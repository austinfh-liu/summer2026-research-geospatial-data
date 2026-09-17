import os
import sys
import json
import csv
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError:
    rasterio = None
    Resampling = None
    reproject = None

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, 'Your_Folder_Name')


# =========================================================================
# 2. HELPER FUNCTIONS
# =========================================================================
def yearly_source_root(year_dir):
    """Return the January source root when present, otherwise the year directory."""
    january_dirs = sorted(
        os.path.join(year_dir, name)
        for name in os.listdir(year_dir)
        if (
            os.path.isdir(os.path.join(year_dir, name))
            and "01_january" in name.strip().casefold()
            and not name.startswith((".", "._"))
        )
    )
    return january_dirs[0] if january_dirs else year_dir


def find_yearly_source_tifs(year_dir):
    """Map each Admin-1 annual dataset to its GeoTIFF and January source root."""
    if not os.path.isdir(year_dir):
        return {}

    source_root = yearly_source_root(year_dir)
    admin1_dirs = sorted(
        (name, os.path.join(source_root, name))
        for name in os.listdir(source_root)
        if (
            os.path.isdir(os.path.join(source_root, name))
            and not name.startswith((".", "._"))
        )
    )
    source_series = {}
    for admin1_label, admin1_dir in admin1_dirs:
        admin1_key = admin1_label.casefold()
        datasets = source_series.setdefault(admin1_key, {})
        for root, dir_names, file_names in os.walk(admin1_dir):
            dir_names[:] = sorted(name for name in dir_names if not name.startswith((".", "._")))
            for file_name in sorted(file_names):
                lower_name = file_name.casefold()
                if not (
                    lower_name.endswith((".tif", ".tiff"))
                    and not file_name.startswith((".", "._"))
                    and "mask" not in lower_name
                    and ".aux" not in lower_name
                ):
                    continue

                tif_path = os.path.join(root, file_name)
                dataset_label = os.path.relpath(tif_path, admin1_dir).replace(os.sep, "/")
                dataset_key = dataset_label.casefold()
                if dataset_key in datasets:
                    print(
                        f"[Warning] Duplicate GeoTIFF dataset {dataset_label} found in Admin-1 "
                        f"{admin1_label} under {source_root}; skipping {tif_path}"
                    )
                    continue
                datasets[dataset_key] = (tif_path, source_root, admin1_label, dataset_label)

        if not datasets:
            del source_series[admin1_key]
    return source_series


def load_tif_safe(file_path):
    """Load a GeoTIFF and normalize NaN, zero, and negative values as missing."""
    if not file_path or not os.path.exists(file_path):
        return None, None, 0
    try:
        with rasterio.open(file_path) as src:
            masked_arr = src.read(1, masked=True).astype(np.float32)
            arr = np.ma.filled(masked_arr, np.nan).astype(np.float32, copy=False)
            profile = src.profile.copy()

        values_to_normalize = (np.isfinite(arr) & (arr <= 0.0)) | np.isinf(arr)
        normalization_count = int(np.count_nonzero(values_to_normalize))
        arr[~np.isfinite(arr) | (arr <= 0.0)] = np.nan
        return arr, profile, normalization_count
    except Exception as error:
        print(f"  [Error] Reading TIF failed ({file_path}): {error}")
        return None, None, 0


def align_for_repair(source_arr, source_profile, target_profile):
    """Align a source raster to a target raster while keeping missing cells as NaN."""
    if source_arr is None or source_profile is None or target_profile is None:
        return None

    target_shape = (target_profile["height"], target_profile["width"])
    same_grid = (
        source_arr.shape == target_shape
        and source_profile.get("crs") == target_profile.get("crs")
        and source_profile.get("transform") == target_profile.get("transform")
    )
    if same_grid:
        return source_arr

    source_transform = source_profile.get("transform")
    target_transform = target_profile.get("transform")
    source_crs = source_profile.get("crs")
    target_crs = target_profile.get("crs")
    if any(value is None for value in (source_transform, target_transform, source_crs, target_crs)):
        print("  [Warning] Cannot align rasters with missing CRS or transform metadata.")
        return None

    alignment_nodata = np.float32(np.finfo(np.float32).min)
    source_values = source_arr.copy()
    source_values[~np.isfinite(source_values)] = alignment_nodata
    aligned = np.full(target_shape, alignment_nodata, dtype=np.float32)
    try:
        reproject(
            source=source_values,
            destination=aligned,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=alignment_nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=alignment_nodata,
            resampling=Resampling.nearest,
        )
    except Exception as error:
        print(f"  [Warning] Reprojection alignment failed: {error}")
        return None

    aligned[aligned == alignment_nodata] = np.nan
    return aligned


def impute_from_temporal_neighbors(
    target_arr,
    target_profile,
    previous_arr,
    previous_profile,
    following_arr,
    following_profile,
    previous_year_distance=1,
    following_year_distance=1,
):
    """Fill missing cells from positive annual neighbors at matching pixel locations."""
    if previous_year_distance <= 0 or following_year_distance <= 0:
        raise ValueError("Temporal source distances must be positive.")

    repaired = target_arr.copy()
    missing = ~np.isfinite(repaired) | (repaired <= 0.0)
    repair_counts = {
        "two_sided_mean_pixels": 0,
        "previous_neighbor_fallback_pixels": 0,
        "following_neighbor_fallback_pixels": 0,
    }
    if not np.any(missing):
        return repaired, repair_counts, "no_missing_pixels"

    previous = align_for_repair(previous_arr, previous_profile, target_profile)
    following = align_for_repair(following_arr, following_profile, target_profile)
    previous_valid = (
        np.isfinite(previous) & (previous > 0.0)
        if previous is not None
        else np.zeros(target_arr.shape, dtype=bool)
    )
    following_valid = (
        np.isfinite(following) & (following > 0.0)
        if following is not None
        else np.zeros(target_arr.shape, dtype=bool)
    )

    two_sided_mask = missing & previous_valid & following_valid
    previous_only_mask = missing & previous_valid & ~following_valid
    following_only_mask = missing & ~previous_valid & following_valid

    if np.any(two_sided_mask):
        total_distance = np.float32(previous_year_distance + following_year_distance)
        previous_weight = np.float32(following_year_distance) / total_distance
        following_weight = np.float32(previous_year_distance) / total_distance
        repaired[two_sided_mask] = (
            previous[two_sided_mask] * previous_weight
            + following[two_sided_mask] * following_weight
        )
        repair_counts["two_sided_mean_pixels"] = int(np.count_nonzero(two_sided_mask))
    if np.any(previous_only_mask):
        repaired[previous_only_mask] = previous[previous_only_mask]
        repair_counts["previous_neighbor_fallback_pixels"] = int(np.count_nonzero(previous_only_mask))
    if np.any(following_only_mask):
        repaired[following_only_mask] = following[following_only_mask]
        repair_counts["following_neighbor_fallback_pixels"] = int(np.count_nonzero(following_only_mask))

    if repair_counts["two_sided_mean_pixels"] and (
        repair_counts["previous_neighbor_fallback_pixels"]
        or repair_counts["following_neighbor_fallback_pixels"]
    ):
        result = "two_sided_mean_and_single_neighbor_fallback"
    elif repair_counts["two_sided_mean_pixels"]:
        result = "two_sided_mean"
    elif repair_counts["previous_neighbor_fallback_pixels"] or repair_counts["following_neighbor_fallback_pixels"]:
        result = "single_neighbor_fallback"
    else:
        result = "no_valid_temporal_source_pixels"
    return repaired, repair_counts, result


def save_tif_safe(file_path, arr, profile):
    """Persist positive values and mark every missing or nonpositive cell as NoData."""
    if profile is None:
        return False

    output_profile = profile.copy()
    out_arr = arr.copy()
    nodata_val = output_profile.get("nodata")
    if nodata_val is None or np.isnan(nodata_val):
        nodata_val = -9999.0

    out_arr[~np.isfinite(out_arr) | (out_arr <= 0.0)] = nodata_val
    output_profile.update(dtype=rasterio.float32, count=1, nodata=nodata_val)
    try:
        with rasterio.open(file_path, "w", **output_profile) as destination:
            destination.write(out_arr.astype(np.float32), 1)
            destination.write_mask(np.where(np.isfinite(arr) & (arr > 0.0), 255, 0).astype(np.uint8))
        return True
    except Exception as error:
        print(f"  [Warning] Could not write GeoTIFF {file_path}: {error}")
        return False


def find_json_sidecar(tif_path, source_root):
    """Find a TIFF JSON sidecar between its directory and the annual source root."""
    direct_match = os.path.splitext(tif_path)[0] + ".json"
    if os.path.isfile(direct_match):
        return direct_match

    current_dir = os.path.dirname(tif_path)
    source_dir = os.path.abspath(source_root)
    while True:
        try:
            json_paths = sorted(
                os.path.join(current_dir, name)
                for name in os.listdir(current_dir)
                if name.casefold().endswith(".json") and not name.startswith((".", "._"))
            )
        except OSError:
            return None

        stem_matches = [
            json_path
            for json_path in json_paths
            if os.path.splitext(os.path.basename(json_path))[0].casefold()
            == os.path.splitext(os.path.basename(tif_path))[0].casefold()
        ]
        if stem_matches:
            return stem_matches[0]
        if len(json_paths) == 1:
            return json_paths[0]
        if os.path.abspath(current_dir) == source_dir:
            return None
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            return None
        current_dir = parent


def update_json_statistics(tif_path, source_root, original_arr, repaired_arr, repair_counts):
    """Store only the requested repair statistics in the JSON sidecar."""
    try:
        json_path = find_json_sidecar(tif_path, source_root)
        if json_path is None:
            print(f"  [Warning] No JSON sidecar found for {tif_path}")
            return False

        with open(json_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("top-level JSON value is not an object")

        original_nan = ~np.isfinite(original_arr)
        repaired_nan = ~np.isfinite(repaired_arr)
        total_pixels = int(repaired_arr.size)
        repaired_pixels = int(sum(repair_counts.values()))
        valid_values = repaired_arr[np.isfinite(repaired_arr)]

        document["stats"] = {
            "min": float(np.min(valid_values)) if valid_values.size else None,
            "max": float(np.max(valid_values)) if valid_values.size else None,
            "mean": float(np.mean(valid_values)) if valid_values.size else None,
            "median": float(np.median(valid_values)) if valid_values.size else None,
            "std": float(np.std(valid_values)) if valid_values.size else None,
            "repaired_pixels": repaired_pixels,
            "nan_percentage_before": float(np.count_nonzero(original_nan) / total_pixels * 100.0),
            "nan_percentage_after": float(np.count_nonzero(repaired_nan) / total_pixels * 100.0),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"  [Warning] Could not update JSON statistics for {tif_path}: {error}")
        return False


def nearest_available_year(year, direction, available_years):
    """Return the closest source year in one temporal direction, or None."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    candidates = [candidate for candidate in available_years if candidate < year] if direction == -1 else [
        candidate for candidate in available_years if candidate > year
    ]
    if not candidates:
        return None
    return str(max(candidates) if direction == -1 else min(candidates))


def year_distance(start_year, end_year):
    """Return the positive number of calendar years from start to end."""
    distance = end_year - start_year
    if distance <= 0:
        raise ValueError("End year must be after start year.")
    return distance


def normalize_country_name(country_value):
    """Remove trailing monthly/yearly suffixes from a country label."""
    country = str(country_value or "").strip()
    lowered = country.casefold()
    for suffix in (" yearly", " monthly"):
        if lowered.endswith(suffix):
            return country[: -len(suffix)].strip()
    return country


def export_yearly_data_json_to_csv():
    """Create the yearly summary CSV from every readable JSON sidecar."""
    output_path = os.path.join(script_dir, "admin1_yearly_summary.csv")
    temporary_output_path = f"{output_path}.tmp"
    field_names = [
        "country",
        "year",
        "iso3",
        "gid_1",
        "admin1_name",
        "shape",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "repaired_pixels",
        "nan_percentage_before",
        "nan_percentage_after",
    ]
    json_paths = []
    for root, dir_names, file_names in os.walk(parent_dir):
        dir_names[:] = sorted(name for name in dir_names if not name.startswith((".", "._")))
        json_paths.extend(
            os.path.join(root, name)
            for name in sorted(file_names)
            if name.casefold().endswith(".json") and not name.startswith((".", "._"))
        )

    rows = []
    for json_path in sorted(json_paths):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            if not isinstance(document, dict):
                raise ValueError("top-level JSON value is not an object")

            relative_parts = os.path.relpath(json_path, parent_dir).split(os.sep)
            year = int(relative_parts[1]) if len(relative_parts) > 1 and relative_parts[1].isdigit() else ""
            stats = document.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}
            shape = document.get("shape", "")
            source_country = document.get("country", relative_parts[0] if relative_parts else "")
            rows.append(
                {
                    "country": normalize_country_name(source_country),
                    "year": year,
                    "iso3": document.get("iso3", ""),
                    "gid_1": document.get("gid_1", ""),
                    "admin1_name": document.get("name_1", document.get("admin1_name", "")),
                    "shape": json.dumps(shape, sort_keys=True) if isinstance(shape, (dict, list)) else shape,
                    "min": stats.get("min", ""),
                    "max": stats.get("max", ""),
                    "mean": stats.get("mean", ""),
                    "median": stats.get("median", ""),
                    "std": stats.get("std", ""),
                    "repaired_pixels": stats.get("repaired_pixels", ""),
                    "nan_percentage_before": stats.get("nan_percentage_before", ""),
                    "nan_percentage_after": stats.get("nan_percentage_after", ""),
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            print(f"[Warning] Skipping JSON file {json_path}: {error}")

    try:
        with open(temporary_output_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_output_path, output_path)
    except OSError as error:
        if os.path.exists(temporary_output_path):
            os.remove(temporary_output_path)
        print(f"[Error] Could not create CSV {output_path}: {error}")
        return False

    if rows:
        print(f"[Info] Exported {len(rows):,} yearly JSON record(s) to {output_path}")
    else:
        print(f"[Info] No JSON data found; created a header-only CSV at {output_path}")
    return True


# =========================================================================
# 3. PROCESSING PIPELINE
# =========================================================================
def main():
    print("Starting Yearly NTL GeoTIFF Imputation Pipeline...")
    print(f"Target Directory: {parent_dir}\n")
    if rasterio is None:
        print("[Error] 'rasterio' package is required for imputation. Install with: pip install rasterio")
        return
    if not os.path.isdir(parent_dir):
        print(f"[Error] Directory does not exist: {parent_dir}")
        return

    country_folders = sorted(
        folder_name
        for folder_name in os.listdir(parent_dir)
        if (
            os.path.isdir(os.path.join(parent_dir, folder_name))
            and folder_name.casefold().endswith(" yearly")
            and not folder_name.startswith((".", "._"))
        )
    )
    overall = {
        "rasters": 0,
        "missing_pixels": 0,
        "repaired_pixels": 0,
        "max_nan_repair_percentage": 0.0,
        "max_nan_repair_pixels": 0,
        "max_nan_initial_pixels": 0,
        "max_nan_repair_tif": None,
    }

    for folder_name in country_folders:
        base_dir = os.path.join(parent_dir, folder_name)
        years = sorted(
            (
                name
                for name in os.listdir(base_dir)
                if name.isdigit() and os.path.isdir(os.path.join(base_dir, name))
            ),
            key=int,
        )
        if not years:
            print(f"[Warning] No year folders found for {folder_name}.")
            continue

        source_series = {
            year: find_yearly_source_tifs(os.path.join(base_dir, year))
            for year in years
        }
        target_records = [
            (year, admin1_key, dataset_key)
            for year in years
            for admin1_key in sorted(source_series[year])
            for dataset_key in sorted(source_series[year][admin1_key])
        ]
        available_years_by_series = {}
        for year in years:
            for admin1_key, datasets in source_series[year].items():
                for dataset_key in datasets:
                    available_years_by_series.setdefault((admin1_key, dataset_key), []).append(int(year))

        print("\n==================================================")
        print(f"PROCESSING COUNTRY: {folder_name}")
        print(f"Years: {years}")
        print("==================================================")
        if not target_records:
            print("  [Warning] No usable GeoTIFFs found.")
            continue

        raw_cache = {}

        def get_raw_raster(year, admin1_key, dataset_key):
            cache_key = (year, admin1_key, dataset_key)
            if cache_key not in raw_cache:
                source = source_series.get(year, {}).get(admin1_key, {}).get(dataset_key)
                if source is None:
                    raw_cache[cache_key] = None
                else:
                    tif_path, source_root, admin1_label, dataset_label = source
                    array, profile, normalization_count = load_tif_safe(tif_path)
                    raw_cache[cache_key] = (
                        (
                            tif_path,
                            source_root,
                            array,
                            profile,
                            admin1_label,
                            dataset_label,
                            normalization_count,
                        )
                        if array is not None and profile is not None
                        else None
                    )
            return raw_cache[cache_key]

        for year, admin1_key, dataset_key in target_records:
            target = get_raw_raster(year, admin1_key, dataset_key)
            if target is None:
                continue

            (
                tif_path,
                source_root,
                target_arr,
                target_profile,
                admin1_label,
                dataset_label,
                normalization_count,
            ) = target
            overall["rasters"] += 1
            missing_mask = ~np.isfinite(target_arr) | (target_arr <= 0.0)
            missing_pixels = int(np.count_nonzero(missing_mask))
            overall["missing_pixels"] += missing_pixels

            available_years = available_years_by_series.get((admin1_key, dataset_key), [])
            previous_year = nearest_available_year(int(year), -1, available_years)
            following_year = nearest_available_year(int(year), 1, available_years)
            previous = (
                get_raw_raster(previous_year, admin1_key, dataset_key) if previous_year else None
            )
            following = (
                get_raw_raster(following_year, admin1_key, dataset_key) if following_year else None
            )

            previous_arr = previous_profile = following_arr = following_profile = None
            if previous is not None:
                _, _, previous_arr, previous_profile, _, _, _ = previous
            if following is not None:
                _, _, following_arr, following_profile, _, _, _ = following

            previous_year_distance = (
                year_distance(int(previous_year), int(year)) if previous_year else 1
            )
            following_year_distance = (
                year_distance(int(year), int(following_year)) if following_year else 1
            )
            repaired_arr, repair_counts, result = impute_from_temporal_neighbors(
                target_arr,
                target_profile,
                previous_arr,
                previous_profile,
                following_arr,
                following_profile,
                previous_year_distance,
                following_year_distance,
            )
            repaired_pixels = int(sum(repair_counts.values()))
            initial_nan_pixels = int(np.count_nonzero(~np.isfinite(target_arr)))
            remaining_nan_pixels = int(np.count_nonzero(~np.isfinite(repaired_arr)))
            nan_repaired_pixels = max(0, initial_nan_pixels - remaining_nan_pixels)
            nan_repair_percentage = (
                (nan_repaired_pixels / initial_nan_pixels) * 100.0
                if initial_nan_pixels > 0
                else 0.0
            )
            if nan_repair_percentage > overall["max_nan_repair_percentage"]:
                overall["max_nan_repair_percentage"] = nan_repair_percentage
                overall["max_nan_repair_pixels"] = nan_repaired_pixels
                overall["max_nan_initial_pixels"] = initial_nan_pixels
                overall["max_nan_repair_tif"] = os.path.relpath(tif_path, parent_dir)

            if repaired_pixels:
                overall["repaired_pixels"] += repaired_pixels
                source_years = f"{previous_year or 'none'} and {following_year or 'none'}"
                print(
                    f"  -> Year {year}: fixed {repaired_pixels:,} pixels "
                    f"using {source_years} [Admin-1: {admin1_label}; dataset: {dataset_label}]."
                )
            elif missing_pixels:
                print(
                    f"  -> Year {year}: no positive temporal source was available ({result}) "
                    f"[Admin-1: {admin1_label}; dataset: {dataset_label}]."
                )

            tif_updated = True
            if repaired_pixels or normalization_count:
                tif_updated = save_tif_safe(tif_path, repaired_arr, target_profile)
            if tif_updated:
                update_json_statistics(
                    tif_path,
                    source_root,
                    target_arr,
                    repaired_arr,
                    repair_counts,
                )

    print("\nProcessing complete!")
    print(f"  GeoTIFFs inspected: {overall['rasters']:,}")
    print(f"  Missing pixels inspected: {overall['missing_pixels']:,}")
    print(f"  Pixels repaired: {overall['repaired_pixels']:,}")
    if overall["max_nan_repair_tif"] is not None:
        print(
            "  Largest NaN repair % in one GeoTIFF: "
            f"{overall['max_nan_repair_percentage']:.2f}% "
            f"({overall['max_nan_repair_pixels']:,}/{overall['max_nan_initial_pixels']:,} NaN pixels) "
            f"- {overall['max_nan_repair_tif']}"
        )
    else:
        print("  Largest NaN repair % in one GeoTIFF: 0.00% (no NaN pixels were repaired).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n[Info] Cancelled by user.")
    finally:
        export_yearly_data_json_to_csv()
