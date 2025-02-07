import tarfile
import pandas as pd
import os
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import tempfile
import logging
import json
import time
from datetime import datetime

import os
os.path.dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DISABLE_PARAMS = True
SKIP_VALIDATION = False
TEST_PROBLEMS = ['p00000', 'p00001']  # Test with just these problems when skipping validation

KAGGLE_PATHS = {
    'input_dir': '/kaggle/input',
    'working_dir': '/kaggle/working',
    'temp_dir': '/kaggle/working/temp',
    'output_dir': '/kaggle/working/output'
}
LOCAL_PATHS = {
    'input_dir': '/Users/jacobfeng/Downloads',
    'working_dir': './output',
    'temp_dir': tempfile.mkdtemp(),
    'output_dir': './output'
}
PARAMS_CONFIG = {
    'codenet_tar': 'Project_CodeNet.tar',
    'metadata_tar': 'Project_CodeNet_metadata.tar',
    'num_pairs': 17240,
    'pairs_per_problem': 8,
    'paths': LOCAL_PATHS,
}


class EnvConfig:
    @staticmethod
    def get_paths():
        return PARAMS_CONFIG['paths']


class CodeNetExtractor:
    def __init__(self, codenet_tar_path: str, metadata_tar_path: str):
        """Initialize with paths to CodeNet archives"""
        self.paths = EnvConfig.get_paths()
        self.start_time = time.time()

        # Full paths to tar files
        self.codenet_tar_path = os.path.join(self.paths['input_dir'], codenet_tar_path)
        self.metadata_tar_path = os.path.join(self.paths['input_dir'], metadata_tar_path)

        os.makedirs(self.paths['temp_dir'], exist_ok=True)
        os.makedirs(self.paths['output_dir'], exist_ok=True)

        # We'll keep references to open tar files here
        self._codenet_tar = None
        self._metadata_tar = None

        logger.info(f"Initialized extractor")
        logger.info(f"Using temporary directory: {self.paths['temp_dir']}")
        logger.info(f"Using output directory: {self.paths['output_dir']}")

    def open_tars(self):
        """
        Open both the CodeNet and metadata tar files in 'r:' mode
        (which is ideal for plain/uncompressed .tar files).
        """
        if self._codenet_tar is None:
            logger.info("Opening CodeNet tar...")
            self._codenet_tar = tarfile.open(self.codenet_tar_path, 'r:')
            logger.info("CodeNet tar opened")

        if self._metadata_tar is None:
            logger.info("Opening metadata tar...")
            self._metadata_tar = tarfile.open(self.metadata_tar_path, 'r:')
            logger.info("Metadata tar opened")

    def close_tars(self):
        """Close the open tar files (if they are open)."""
        if self._codenet_tar is not None:
            logger.info("Closing CodeNet tar...")
            self._codenet_tar.close()
            self._codenet_tar = None
            logger.info("CodeNet tar closed")

        if self._metadata_tar is not None:
            logger.info("Closing metadata tar...")
            self._metadata_tar.close()
            self._metadata_tar = None
            logger.info("Metadata tar closed")

    def log_progress(self, message: str, start_time: float = None):
        """Log progress with elapsed time"""
        if start_time is None:
            start_time = self.start_time
        elapsed = time.time() - start_time
        logger.info(f"{message} (Elapsed: {elapsed:.2f}s)")

    def extract_problem_list(self) -> pd.DataFrame:
        """Extract and read the problem_list.csv from metadata archive"""
        start_time = time.time()
        logger.info("Extracting problem list from metadata archive...")

        # Ensure metadata tar is open
        self.open_tars()

        problem_list_file = self._metadata_tar.extractfile('Project_CodeNet/metadata/problem_list.csv')
        if problem_list_file is None:
            raise FileNotFoundError("problem_list.csv not found in metadata archive")

        df = pd.read_csv(problem_list_file)
        self.log_progress(f"Successfully extracted problem list with {len(df)} problems", start_time)
        return df

    def extract_problem_metadata(self, problem_id: str) -> pd.DataFrame:
        """Extract and read metadata for a specific problem from metadata tar"""
        self.open_tars()  # Ensure metadata tar is open

        metadata_file = self._metadata_tar.extractfile(f'Project_CodeNet/metadata/{problem_id}.csv')
        if metadata_file is None:
            raise FileNotFoundError(f"Metadata for problem {problem_id} not found")
        return pd.read_csv(metadata_file)

    def extract_code_pair(self, problem_id: str, submission_id: str, language: str) -> str:
        """Extract source code for a specific submission from the CodeNet tar"""
        # Make sure the CodeNet tar is open
        self.open_tars()

        logger.info(f"\tExtracting {language} submission {submission_id}")

        # Determine the file extension by language
        ext = 'py' if language == 'Python' else 'java'
        file_path = f'Project_CodeNet/data/{problem_id}/{language}/{submission_id}.{ext}'

        try:
            code_file = self._codenet_tar.extractfile(file_path)
            if code_file is None:
                raise FileNotFoundError(f"Source file not found: {file_path}")
            logger.info(f"\t\t'tar.extractfile()' success")
            return code_file.read().decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error extracting {file_path}: {str(e)}")
            return None

    def get_valid_problems(self, min_submissions: int = 5) -> List[str]:
        """
        Find problems with sufficient accepted submissions in both Java and Python
        by reading the metadata tar once for each problem.
        """
        start_time = time.time()
        logger.info("Starting valid problem search...")

        problems = self.extract_problem_list()
        valid_problems = []
        total = len(problems)

        for idx, problem in problems.iterrows():
            problem_id = problem['id']
            try:
                if idx % 100 == 0:
                    self.log_progress(f"Processing problem {idx}/{total}")

                metadata = self.extract_problem_metadata(problem_id)

                # Count accepted submissions for each language
                accepted = metadata[metadata['status'] == 'Accepted']
                java_count = len(accepted[accepted['language'] == 'Java'])
                python_count = len(accepted[accepted['language'] == 'Python'])

                if java_count >= min_submissions and python_count >= min_submissions:
                    valid_problems.append(problem_id)
                    logger.info(f"✓ Problem {problem_id} qualifies with Java:{java_count}, Python:{python_count}")

            except Exception as e:
                logger.warning(f"Error processing problem {problem_id}: {str(e)}")
                continue

        self.log_progress(f"Valid problem search completed. Found {len(valid_problems)} valid problems", start_time)
        return valid_problems

    def create_dataset(self, output_dir: str, num_pairs: int = 10000, pairs_per_problem: int = 5):
        """Create the final dataset of Java-Python pairs"""
        dataset_start = time.time()
        logger.info(f"Starting dataset creation. Target: {num_pairs} pairs ({pairs_per_problem} per problem)")

        # Make sure the tars are open
        self.open_tars()

        os.makedirs(output_dir, exist_ok=True)
        if SKIP_VALIDATION:
            logger.info("⚠️ SKIP_VALIDATION is True - using test problems list")
            valid_problems = TEST_PROBLEMS
        else:
            valid_problems = self.get_valid_problems()

        logger.info(f"Beginning extraction from {len(valid_problems)} valid problems...")
        pairs_extracted = 0
        problems_processed = 0
        dataset = []

        for problem_id in valid_problems:
            problems_processed += 1
            if pairs_extracted >= num_pairs:
                break

            try:
                extraction_start = time.time()
                logger.info(f"Processing problem {problem_id} ({problems_processed}/{len(valid_problems)})")

                metadata = self.extract_problem_metadata(problem_id)
                accepted = metadata[metadata['status'] == 'Accepted']

                # Get submissions for both languages
                java_submissions = (
                    accepted[accepted['language'] == 'Java']
                    .sort_values('cpu_time')['submission_id']
                    .tolist()
                )
                python_submissions = (
                    accepted[accepted['language'] == 'Python']
                    .sort_values('cpu_time')['submission_id']
                    .tolist()
                )

                logger.info(
                    f"Found {len(java_submissions)} Java and {len(python_submissions)} Python submissions for {problem_id}")

                # Extract up to pairs_per_problem pairs
                pairs_before = pairs_extracted
                for i in range(min(pairs_per_problem, len(java_submissions), len(python_submissions))):
                    pair_start = time.time()
                    logger.info(f"\tExtracting pair {i + 1}/{pairs_per_problem} for problem {problem_id}")

                    java_code = self.extract_code_pair(problem_id, java_submissions[i], 'Java')
                    python_code = self.extract_code_pair(problem_id, python_submissions[i], 'Python')

                    if java_code and python_code:
                        pair_data = {
                            'problem_id': problem_id,
                            'java_submission_id': java_submissions[i],
                            'python_submission_id': python_submissions[i],
                            'java_code': java_code,
                            'python_code': python_code
                        }
                        dataset.append(pair_data)
                        pairs_extracted += 1

                        self.log_progress(f"\tSuccessfully extracted pair {i + 1} for {problem_id}", pair_start)

                        if pairs_extracted % 5000 == 0:
                            save_start = time.time()
                            intermediate_path = os.path.join(
                                output_dir,
                                f'java_python_pairs_intermediate_{pairs_extracted}.json'
                            )
                            with open(intermediate_path, 'w') as f:
                                json.dump(dataset, f, indent=2)
                            self.log_progress(f"Saved intermediate checkpoint at {pairs_extracted} pairs", save_start)

                pairs_this_problem = pairs_extracted - pairs_before
                self.log_progress(
                    f"Completed problem {problem_id}: extracted {pairs_this_problem} pairs. "
                    f"Total progress: {pairs_extracted}/{num_pairs} pairs",
                    extraction_start
                )

            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                continue

        # Save the final dataset
        save_start = time.time()
        output_path = os.path.join(output_dir, 'java_python_pairs_final.json')
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        self.log_progress(
            f"Dataset creation completed. Extracted {len(dataset)} pairs to {output_path}",
            dataset_start
        )

        # Optionally close the tars here if you want
        # self.close_tars()


def main():
    start_time = time.time()
    logger.info("Starting CodeNet extraction process...")

    if DISABLE_PARAMS:
        # Direct execution with Kaggle/LOCAL configuration
        extractor = CodeNetExtractor(
            PARAMS_CONFIG['codenet_tar'],
            PARAMS_CONFIG['metadata_tar']
        )
        extractor.open_tars()
        try:
            extractor.create_dataset(
                output_dir='extracted_pairs',
                num_pairs=PARAMS_CONFIG['num_pairs'],
                pairs_per_problem=PARAMS_CONFIG['pairs_per_problem']
            )
        finally:
            # Always close tar files even if there's an exception
            extractor.close_tars()

    else:
        # Command line execution
        parser = argparse.ArgumentParser(description='Extract Java-Python pairs from CodeNet dataset')
        parser.add_argument('--codenet-tar', required=True, help='Path to Project_CodeNet.tar')
        parser.add_argument('--metadata-tar', required=True, help='Path to Project_CodeNet_metadata.tar')
        parser.add_argument('--output-dir', required=True, help='Output directory for extracted pairs')
        parser.add_argument('--num-pairs', type=int, default=10000, help='Number of pairs to extract')
        parser.add_argument('--pairs-per-problem', type=int, default=5, help='Maximum pairs per problem')

        args = parser.parse_args()
        extractor = CodeNetExtractor(args.codenet_tar, args.metadata_tar)
        extractor.open_tars()
        try:
            extractor.create_dataset(args.output_dir, args.num_pairs, args.pairs_per_problem)
        finally:
            extractor.close_tars()

    elapsed = time.time() - start_time
    logger.info(f"Extraction process completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
