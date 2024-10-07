import os
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from image_utils import convert_series_to_nifti  
from tqdm import tqdm

class DICOMSeriesExtractor:
    def __init__(self, desired_series=None, ignore_terms=None):
        self.desired_series = {series.upper().strip() for series in desired_series} if desired_series else set()
        self.ignore_terms = {term.upper().strip() for term in ignore_terms} if ignore_terms else set()
        self.matched_dicom_files = defaultdict(list)

    def process_dicom_file(self, dicom_path):
        """Process a single DICOM file to extract the series description."""
        try:
            dicom = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            series_description = getattr(dicom, 'SeriesDescription', 'Unknown').upper().strip()
            if any(ignore_term in series_description for ignore_term in self.ignore_terms):
                # print(f"Ignore term found in series description '{series_description}' for file '{dicom_path}'")
                return None
            return dicom_path, series_description
        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
            return None

    def validate_dicom_series(self, base_path):
        """Validate DICOM series across all patients in the specified directory."""
        print("Starting validation of DICOM series...")
        for patient_dir in os.listdir(base_path):
            patient_path = os.path.join(base_path, patient_dir)
            if os.path.isdir(patient_path):
                print(f"Processing patient directory: {patient_dir}")
                series_found = set()
                all_series_descriptions = set()
                dicom_files = [os.path.join(root, file) 
                            for root, _, files in os.walk(patient_path) 
                            for file in files if file.lower().endswith('.dcm')]
                for dicom_file in dicom_files:
                    result = self.process_dicom_file(dicom_file)
                    if result:
                        dicom_path, series_description = result
                        series_description_upper = series_description.upper()  # Normalize to upper case for comparison
                        all_series_descriptions.add(series_description)
                        if series_description_upper in self.desired_series:
                            series_found.add(series_description)
                            self.matched_dicom_files[patient_dir].append(dicom_path)

                # Change in printing logic:
                if not series_found:
                    print(f"No desired series identified for patient '{patient_dir}'. All available series: {all_series_descriptions}")
                else:
                    print(f"Validated series for {patient_dir}: {series_found}")
                    #print(f"All available series: {all_series_descriptions}")

        print("Validation complete.")


    def copy_dicom(self, target_base):
        """Convert matched DICOM files to NIfTI and save them directly to the specified directory."""
        print("Starting conversion of DICOM files to NIfTI format...")
        if not os.path.exists(target_base):
            os.makedirs(target_base)
            
        for patient, paths in self.matched_dicom_files.items():
            if paths:  # Ensure there are paths to process
                output_file = os.path.join(target_base, f"{patient}.nii")
                try:
                    # Convert the first DICOM path in the list (assuming single series processed per patient)
                    convert_series_to_nifti(os.path.dirname(paths[0]), output_file)
                    print(paths[0] + " " + output_file)
                    print(f"Converted and saved DICOM from {patient} to {output_file}")
                except Exception as e:
                    print(f"Failed to convert DICOM for {patient}: {e}")
            else:
                print(f"No valid DICOM files found for patient {patient}. No NIfTI file created.")
        print("All DICOM conversions completed.")

    def print_matched_series(self):
        """Print the matched series for each patient."""
        for patient, paths in self.matched_dicom_files.items():
            print(f"Patient: {patient}")
            for path in paths:
                print(f"  DICOM Path: {path}")


class DICOMSeriesManager:
    def __init__(self, base_path):
        self.base_path = base_path
        self.patient_series = defaultdict(set)  # Use a set to automatically handle uniqueness

    def list_dicom_series_by_patient(self):
        """List all DICOM series descriptions organized by patient."""
        for patient_dir in tqdm(os.listdir(self.base_path)):
            patient_path = os.path.join(self.base_path, patient_dir)
            if os.path.isdir(patient_path):
                for root, dirs, files in os.walk(patient_path):
                    for file in files:
                        if file.lower().endswith('.dcm'):
                            dicom_path = os.path.join(root, file)
                            try:
                                dicom = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                                series_description = dicom.get('SeriesDescription', 'No Description')
                                self.patient_series[patient_dir].add(series_description)
                            except Exception as e:
                                print(f"Error processing {dicom_path}: {e}")

        return self.patient_series

    def filter_dicom_series(self, ignore_terms):
        """Remove series containing any of the ignore terms."""
        ignore_terms = [term.upper() for term in ignore_terms]  # Case-insensitive comparison

        for patient in list(self.patient_series.keys()):
            filtered_series = {
                description for description in self.patient_series[patient]
                if not any(ignore_term in description.upper() for ignore_term in ignore_terms)
            }
            self.patient_series[patient] = filtered_series

    def print_series(self):
        """Print the series descriptions for each patient."""
        for patient, series in sorted(self.patient_series.items()):
            print(f"Patient: {patient}")
            for description in sorted(series):  # Sorting for consistent output
                print(f"  Series Description: {description}")