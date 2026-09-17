## Using the Imputation Scripts

Before running the imputation scripts, make sure the following Python packages are installed:

```bash
pip install pandas numpy rasterio
```

### Choosing the Correct Script

I created three separate imputation scripts depending on the type of data being processed:

* **Admin0 Monthly Imputation** – for country-level monthly data
* **Admin1 Monthly Imputation** – for Admin1-level monthly data
* **Yearly Imputation** – for yearly data

### How to Run the Imputation

1. Create a new folder and place the appropriate imputation script inside it.
2. Place your **Clipped TIF data folder** inside the same folder.
3. In the script, locate the `parent_dir` setting near the beginning of the file and change the folder name to match your Clipped TIF folder:

```python
parent_dir = os.path.join(script_dir, 'Your_Folder_Name_Here')
```

4. **Optional:** For countries that are heavily affected by monsoon seasons, you can include a CSV file specifying the heavy monsoon months for each country. Refer to the `monsoon_config` example in this repository for the expected format.
5. Run the script. Once processing is complete, it will generate a single CSV file containing the processed data.

### Output

The final CSV includes the processed observations along with information that can be used to evaluate the data before and after imputation, including:

* `nan_percentage_before`
* `nan_percentage_after`
* Mean
* Median
* Standard deviation
* Additional metadata and summary statistics

The resulting CSV can then be used for further analysis or combined with the processed datasets from the rest of the project.
