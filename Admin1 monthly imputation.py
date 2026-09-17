import os
import sys
import re
import argparse
import json
import pandas as pd
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError:
    print("[Error] 'rasterio' package is required. Install with: pip install rasterio")
    sys.exit(1)

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, 'Monthly 2018 - 2020')
config_path = os.path.join(script_dir, 'monsoon_config.csv')

if not os.path.exists(parent_dir):
    print(f"[Error] Directory does not exist: {parent_dir}")
    sys.exit(1)

monsoon_lookup = {}
csv_found = os.path.exists(config_path)

if csv_found:
    print(f"[Info] Monsoon config CSV detected at: {config_path}")
    try:
        df = pd.read_csv(config_path)
        for _, r in df.iterrows():
            if str(r.get('Has_Monsoon')).lower() == 'true' and pd.notna(r.get('Monsoon_Months')):
                country_name = str(r['Country']).strip().lower()
                months = set(map(int, str(r['Monsoon_Months']).split(',')))
                monsoon_lookup[country_name] = months
    except Exception as e:
        print(f"[Warning] Failed to parse {config_path}: {e}")
        csv_found = False
else:
    print("[Info] No monsoon_config.csv detected. Monthly interpolation will be used for all months.")

# =========================================================================
# 2. HELPER FUNCTIONS
# =========================================================================
def load_tif_safe(file_path):
    """
    Safely opens a GeoTIFF file.
    Returns:
      - arr: float32 numpy array
      - profile: rasterio profile dictionary
    """
    if not file_path or not os.path.exists(file_path):
        return None, None
    try:
        with rasterio.open(file_path) as src:
            masked_arr = src.read(1, masked=True).astype(np.float32)
            arr = np.ma.filled(masked_arr, np.nan).astype(np.float32, copy=False)
            profile = src.profile.copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr, profile
    except Exception as e:
        print(f"[Error] Reading TIF failed ({file_path}): {e}")
        return None, None


def get_month_dirs(y_dir):
    """Returns sorted list of tuples: (month_num, folder_path, folder_name)."""
    if not os.path.exists(y_dir):
        return []
    res = []
    for d in os.listdir(y_dir):
        full_path = os.path.join(y_dir, d)
        if os.path.isdir(full_path):
            m = re.match(r'^(\d+)', d.strip())
            if m and 1 <= int(m.group(1)) <= 12:
                res.append((int(m.group(1)), full_path, d))
    return sorted(res, key=lambda x: x[0])


def get_admin1_dirs(month_folder_path):
    """Return immediate Admin-1 directories within a month directory."""
    if not month_folder_path or not os.path.isdir(month_folder_path):
        return []
    return [
        (name, os.path.join(month_folder_path, name))
        for name in sorted(os.listdir(month_folder_path))
        if os.path.isdir(os.path.join(month_folder_path, name)) and not name.startswith((".", "._"))
    ]


# =========================================================================
# 3. PROCESSING PIPELINE
# =========================================================================
def country_key(country_name):
    """Normalize case and punctuation so folder and CSV country names match."""
    return re.sub(r"[^a-z0-9]+", "", country_name.casefold())


def find_source_tifs(month_folder_path):
    """Map each Admin-1 source series to its single GeoTIFF for a month."""
    if not month_folder_path or not os.path.isdir(month_folder_path):
        return {}

    source_series = {}
    for admin1_name, admin1_dir in get_admin1_dirs(month_folder_path):
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
                nested_series = os.path.relpath(os.path.dirname(tif_path), admin1_dir)
                nested_series_key = "" if nested_series == "." else nested_series.replace(os.sep, "/").casefold()
                series_key = (admin1_name.casefold(), nested_series_key)
                series_label = admin1_name if not nested_series_key else f"{admin1_name}/{nested_series}"
                if series_key in source_series:
                    print(
                        f"[Warning] Multiple GeoTIFFs found in Admin-1 series {series_label} "
                        f"under {month_folder_path}; skipping {os.path.basename(tif_path)}"
                    )
                    continue
                source_series[series_key] = (tif_path, series_label, admin1_dir)
    return source_series


def align_for_repair(source_arr, source_profile, target_profile):
    """Align a source raster to the target while preserving missing values as NaN."""
    if source_arr is None or source_profile is None:
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
        print("[Warning] Cannot align rasters with missing CRS or transform metadata.")
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
        print(f"[Warning] Reprojection alignment failed: {error}")
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
    previous_month_distance=1,
    following_month_distance=1,
):
    """Fill missing target cells from positive temporal values at the same pixel location."""
    if previous_month_distance <= 0 or following_month_distance <= 0:
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
    previous_valid = np.isfinite(previous) & (previous > 0.0) if previous is not None else np.zeros(target_arr.shape, dtype=bool)
    following_valid = np.isfinite(following) & (following > 0.0) if following is not None else np.zeros(target_arr.shape, dtype=bool)

    two_sided_mask = missing & previous_valid & following_valid
    previous_only_mask = missing & previous_valid & ~following_valid
    following_only_mask = missing & ~previous_valid & following_valid

    if np.any(two_sided_mask):
        total_distance = np.float32(previous_month_distance + following_month_distance)
        previous_weight = np.float32(following_month_distance) / total_distance
        following_weight = np.float32(previous_month_distance) / total_distance
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
        repair_counts["previous_neighbor_fallback_pixels"] or repair_counts["following_neighbor_fallback_pixels"]
    ):
        result = "two_sided_mean_and_single_neighbor_fallback"
    elif repair_counts["two_sided_mean_pixels"]:
        result = "two_sided_mean"
    elif repair_counts["previous_neighbor_fallback_pixels"] or repair_counts["following_neighbor_fallback_pixels"]:
        result = "single_neighbor_fallback"
    else:
        result = "no_valid_temporal_source_pixels"
    return repaired, repair_counts, result


def save_repaired_tif(file_path, values, profile):
    """Persist repaired values and mark only positive cells as valid."""
    output_profile = profile.copy()
    nodata_value = output_profile.get("nodata")
    if nodata_value is None or not np.isfinite(nodata_value):
        nodata_value = -9999.0

    output_values = values.copy()
    output_values[~np.isfinite(output_values) | (output_values <= 0.0)] = nodata_value
    output_profile.update(dtype=rasterio.float32, count=1, nodata=float(nodata_value))

    with rasterio.open(file_path, "w", **output_profile) as destination:
        destination.write(output_values.astype(np.float32), 1)
        destination.write_mask(np.where(np.isfinite(values) & (values > 0.0), 255, 0).astype(np.uint8))


def find_json_sidecar(tif_path, admin1_dir):
    """Find the TIFF's JSON sidecar without searching outside its Admin-1 directory."""
    direct_match = os.path.splitext(tif_path)[0] + ".json"
    if os.path.isfile(direct_match):
        return direct_match

    current_dir = os.path.dirname(tif_path)
    admin1_dir = os.path.abspath(admin1_dir)
    while True:
        json_paths = sorted(
            os.path.join(current_dir, name)
            for name in os.listdir(current_dir)
            if name.casefold().endswith(".json") and not name.startswith((".", "._"))
        )
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
        if os.path.abspath(current_dir) == admin1_dir:
            return None
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            return None
        current_dir = parent


def update_json_statistics(tif_path, admin1_dir, original_arr, repaired_arr, repair_counts):
    """Store only the requested repair statistics in the JSON sidecar."""
    json_path = find_json_sidecar(tif_path, admin1_dir)
    if json_path is None:
        print(f"[Warning] No JSON sidecar found for {tif_path}")
        return False

    try:
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
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "mean": float(np.mean(valid_values)),
            "median": float(np.median(valid_values)),
            "std": float(np.std(valid_values)),
            "repaired_pixels": repaired_pixels,
            "nan_percentage_before": float(np.count_nonzero(original_nan) / total_pixels * 100.0),
            "nan_percentage_after": float(np.count_nonzero(repaired_nan) / total_pixels * 100.0),
        }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"[Warning] Could not update JSON statistics for {tif_path}: {error}")
        return False


def collect_json_summary_rows(year_dir, country, year):
    """Collect one flattened summary row per JSON sidecar in a country-year directory."""
    rows = []
    for root, dir_names, file_names in os.walk(year_dir):
        dir_names[:] = sorted(name for name in dir_names if not name.startswith((".", "._")))
        for file_name in sorted(file_names):
            if not file_name.casefold().endswith(".json") or file_name.startswith((".", "._")):
                continue

            json_path = os.path.join(root, file_name)
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    document = json.load(handle)
                if not isinstance(document, dict):
                    raise ValueError("top-level JSON value is not an object")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"[Warning] Could not include JSON file in summary ({json_path}): {error}")
                continue

            relative_path = os.path.relpath(json_path, year_dir)
            month_match = re.match(r"^(\d+)", relative_path)
            row = {
                "country": country,
                "year": int(year),
                "month": int(month_match.group(1)) if month_match else None,
                "source_file": relative_path.replace(os.sep, "/"),
            }
            for key, value in document.items():
                if key == "stats" and isinstance(value, dict):
                    row.update(value)
                elif key == "country":
                    continue
                elif key.casefold().replace("_", " ") == "time created":
                    continue
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
                else:
                    row[key] = json.dumps(value, sort_keys=True)
            rows.append(row)

    if not rows:
        print(f"[Warning] No valid JSON files found for summary: {year_dir}")
    return rows

def write_combined_summary_csv(rows):
    """Write all JSON sidecar statistics to one CSV beside this script."""
    summary_path = os.path.join(script_dir, "admin1_monthly_summary.csv")
    if not rows:
        print("[Warning] No valid JSON files found for the combined summary.")
        return None

    summary_frame = pd.DataFrame(rows).rename(columns={"name_1": "admin1_name"})
    summary_frame = summary_frame.drop(
        columns=["source_file", "name", "crs", "created"], errors="ignore"
    )
    summary_frame.sort_values(["year", "month"], na_position="last").to_csv(
        summary_path, index=False
    )
    print(f"  -> Wrote {len(rows):,} JSON records to {summary_path}")
    return summary_path


def calendar_neighbor(year, month, direction):
    """Return the immediately previous or following calendar month, crossing years."""
    if direction == -1:
        return (year - 1, 12) if month == 1 else (year, month - 1)
    if direction == 1:
        return (year + 1, 1) if month == 12 else (year, month + 1)
    raise ValueError("direction must be -1 or 1")


def month_distance(start_year, start_month, end_year, end_month):
    """Return the positive number of calendar months from start to end."""
    distance = (end_year - start_year) * 12 + (end_month - start_month)
    if distance <= 0:
        raise ValueError("End month must be after start month.")
    return distance


def nearest_available_month(year, month, direction, available_months):
    """Return the closest available month in one temporal direction, or None."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    first_year = min(map(int, available_months))
    last_year = max(map(int, available_months))
    candidate_year, candidate_month = calendar_neighbor(year, month, direction)
    while first_year <= candidate_year <= last_year:
        if candidate_month in available_months.get(str(candidate_year), set()):
            return str(candidate_year), candidate_month
        candidate_year, candidate_month = calendar_neighbor(candidate_year, candidate_month, direction)
    return None


def run_self_test():
    """Verify two-sided means and one-sided fallbacks use only matching pixel locations."""
    target = np.array([[np.nan, np.nan, np.nan, 0.0], [np.nan, np.nan, 7.0, np.nan]], dtype=np.float32)
    previous = np.array([[2.0, 2.0, np.nan, 4.0], [np.nan, 5.0, 5.0, np.nan]], dtype=np.float32)
    following = np.array([[4.0, np.nan, 8.0, 8.0], [6.0, np.nan, 9.0, np.nan]], dtype=np.float32)
    profile = {"height": 2, "width": 4, "crs": None, "transform": None}

    repaired, repair_counts, result = impute_from_temporal_neighbors(
        target, profile, previous, profile, following, profile
    )

    assert result == "two_sided_mean_and_single_neighbor_fallback"
    assert repair_counts == {
        "two_sided_mean_pixels": 2,
        "previous_neighbor_fallback_pixels": 2,
        "following_neighbor_fallback_pixels": 2,
    }
    assert repaired[0, 0] == 3.0
    assert repaired[0, 1] == 2.0
    assert repaired[0, 2] == 8.0
    assert repaired[0, 3] == 6.0
    assert repaired[1, 0] == 6.0
    assert repaired[1, 1] == 5.0
    assert repaired[1, 2] == 7.0
    assert np.isnan(repaired[1, 3])

    one_source_target = np.array([[np.nan]], dtype=np.float32)
    one_source_repaired, one_source_counts, one_source_result = impute_from_temporal_neighbors(
        one_source_target, {"height": 1, "width": 1, "crs": None, "transform": None},
        np.array([[9.0]], dtype=np.float32), {"height": 1, "width": 1, "crs": None, "transform": None},
        None, None,
    )
    assert one_source_result == "single_neighbor_fallback"
    assert one_source_repaired[0, 0] == 9.0
    assert one_source_counts["previous_neighbor_fallback_pixels"] == 1
    weighted_repaired, _, _ = impute_from_temporal_neighbors(
        np.array([[np.nan]], dtype=np.float32),
        {"height": 1, "width": 1, "crs": None, "transform": None},
        np.array([[2.0]], dtype=np.float32), {"height": 1, "width": 1, "crs": None, "transform": None},
        np.array([[8.0]], dtype=np.float32), {"height": 1, "width": 1, "crs": None, "transform": None},
        previous_month_distance=1,
        following_month_distance=2,
    )
    assert np.isclose(weighted_repaired[0, 0], 4.0)
    assert calendar_neighbor(2019, 1, -1) == (2018, 12)
    assert calendar_neighbor(2019, 12, 1) == (2020, 1)
    available_months = {"2018": {1, 3}, "2019": {1}}
    assert nearest_available_month(2018, 1, 1, available_months) == ("2018", 3)
    assert nearest_available_month(2019, 1, -1, available_months) == ("2018", 3)
    assert nearest_available_month(2018, 1, -1, available_months) is None
    assert month_distance(2018, 2, 2018, 5) == 3
    assert country_key("B & H") == country_key("B&H")
    print("Self-test passed: same-location two-sided means and one-sided fallbacks work correctly.")


def build_parser():
    parser = argparse.ArgumentParser(description="Monthly GeoTIFF imputation with two-sided means and one-neighbor fallbacks.")
    parser.add_argument("--input-dir", default=parent_dir, help="Root directory containing '<country> Monthly' folders.")
    parser.add_argument("--country", help="Process one country folder name, ignoring punctuation and case.")
    parser.add_argument("--dry-run", action="store_true", help="Report expected repairs without changing TIFF or JSON files.")
    parser.add_argument("--monthly-only", action="store_true", help="Ignore the monsoon CSV and use monthly averaging for every month.")
    parser.add_argument("--self-test", action="store_true", help="Run the in-memory interpolation test and exit.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"[Error] Directory does not exist: {input_dir}")
        sys.exit(1)

    effective_csv_found = csv_found and not args.monthly_only
    normalized_monsoon_lookup = {}
    if effective_csv_found:
        for configured_country, configured_months in monsoon_lookup.items():
            normalized_monsoon_lookup.setdefault(country_key(configured_country), set()).update(configured_months)

    if args.dry_run:
        print("[Info] Dry run only. No GeoTIFF or JSON files will be changed.")
    else:
        print("[Info] Updating GeoTIFFs and JSON sidecars in place.")

    requested_country = country_key(args.country) if args.country else None
    overall = {
        "rasters": 0,
        "missing_pixels": 0,
        "repaired_pixels": 0,
        "max_nan_repair_percentage": 0.0,
        "max_nan_repair_pixels": 0,
        "max_nan_initial_pixels": 0,
        "max_nan_repair_tif": None,
    }
    processed_countries = 0
    summary_rows = []

    for folder in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if not (os.path.isdir(folder_path) and folder.casefold().endswith(" monthly") and not folder.startswith((".", "._"))):
            continue

        country = re.sub(r"\s+monthly$", "", folder, flags=re.IGNORECASE).strip()
        if requested_country and country_key(country) != requested_country:
            continue

        years = sorted(
            [name for name in os.listdir(folder_path) if name.isdigit() and os.path.isdir(os.path.join(folder_path, name))],
            key=int,
        )
        month_dirs = {
            year: {month: path for month, path, _ in get_month_dirs(os.path.join(folder_path, year))}
            for year in years
        }
        available_months = {year: set(month_dirs[year]) for year in years}
        source_series = {
            year: {
                month: find_source_tifs(month_folder_path)
                for month, month_folder_path in month_dirs[year].items()
            }
            for year in years
        }
        target_records = [
            (year, month, series_key)
            for year in years
            for month in sorted(month_dirs[year])
            for series_key in sorted(source_series[year][month])
        ]
        raw_cache = {}

        def get_raw_raster(year, month, series_key):
            key = (year, month, series_key)
            if key not in raw_cache:
                source = source_series.get(year, {}).get(month, {}).get(series_key)
                if source is None:
                    raw_cache[key] = None
                else:
                    tif_path, series_label, admin1_dir = source
                    array, profile = load_tif_safe(tif_path)
                    raw_cache[key] = (
                        (tif_path, admin1_dir, array, profile, series_label)
                        if array is not None and profile is not None
                        else None
                    )
            return raw_cache[key]

        monsoon_months = normalized_monsoon_lookup.get(country_key(country), set()) if effective_csv_found else set()
        print("\n==================================================")
        print(f"PROCESSING COUNTRY: {country}")
        if effective_csv_found:
            print(f"Monsoon months using cross-year averaging: {sorted(monsoon_months) or 'None'}")
        else:
            print("Mode: calendar-month averaging for all months")
        print(f"Years: {years}")
        print("==================================================")

        for year, month, series_key in target_records:
                target = get_raw_raster(year, month, series_key)
                if target is None:
                    continue

                tif_path, admin1_dir, target_arr, target_profile, series_label = target
                overall["rasters"] += 1
                overall["missing_pixels"] += int(np.count_nonzero(~np.isfinite(target_arr) | (target_arr <= 0.0)))

                if effective_csv_found and month in monsoon_months:
                    previous_key = (str(int(year) - 1), month)
                    following_key = (str(int(year) + 1), month)
                    base_method = "cross_year_same_month_mean"
                else:
                    previous_key = nearest_available_month(int(year), month, -1, available_months)
                    following_key = nearest_available_month(int(year), month, 1, available_months)
                    base_method = "calendar_month_mean"

                previous = get_raw_raster(previous_key[0], previous_key[1], series_key) if previous_key else None
                following = get_raw_raster(following_key[0], following_key[1], series_key) if following_key else None
                source_dates = [
                    f"{previous_key[0]}-{previous_key[1]:02d}" if previous_key else "none",
                    f"{following_key[0]}-{following_key[1]:02d}" if following_key else "none",
                ]

                previous_arr = previous_profile = following_arr = following_profile = None
                if previous is not None:
                    _, _, previous_arr, previous_profile, _ = previous
                if following is not None:
                    _, _, following_arr, following_profile, _ = following
                previous_month_distance = (
                    month_distance(int(previous_key[0]), previous_key[1], int(year), month)
                    if previous_key
                    else 1
                )
                following_month_distance = (
                    month_distance(int(year), month, int(following_key[0]), following_key[1])
                    if following_key
                    else 1
                )
                repaired_arr, repair_counts, _ = impute_from_temporal_neighbors(
                    target_arr,
                    target_profile,
                    previous_arr,
                    previous_profile,
                    following_arr,
                    following_profile,
                    previous_month_distance,
                    following_month_distance,
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
                    overall["max_nan_repair_tif"] = os.path.relpath(tif_path, input_dir)
                if repaired_pixels:
                    overall["repaired_pixels"] += repaired_pixels
                    series_suffix = f" [{series_label}]" if len(source_series[year][month]) > 1 else ""
                    print(
                        f"  -> {year}-M{month:02d}: fixed {repaired_pixels:,} pixels "
                        f"using {source_dates[0]} and {source_dates[1]} ({base_method}){series_suffix}"
                    )

                if not args.dry_run:
                    if repaired_pixels:
                        save_repaired_tif(tif_path, repaired_arr, target_profile)
                    update_json_statistics(
                        tif_path,
                        admin1_dir,
                        target_arr,
                        repaired_arr,
                        repair_counts,
                    )

        if not args.dry_run:
            for year in years:
                summary_rows.extend(
                    collect_json_summary_rows(os.path.join(folder_path, year), country, year)
                )

        processed_countries += 1

    if requested_country and processed_countries == 0:
        print(f"[Error] Country not found: {args.country}")
        sys.exit(1)

    if not args.dry_run:
        write_combined_summary_csv(summary_rows)

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
