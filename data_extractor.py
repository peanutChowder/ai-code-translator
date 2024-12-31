import tarfile
import pandas as pd
import os
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import tempfile
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISABLE_PARAMS = True

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
    'num_pairs': 6,
    'pairs_per_problem': 3,
    'paths': LOCAL_PATHS
}

class EnvConfig:
    @staticmethod
    def get_paths():
        return PARAMS_CONFIG['paths']


class CodeNetExtractor:
    def __init__(self, codenet_tar_path: str, metadata_tar_path: str):
        """Initialize with paths to CodeNet archives"""
        self.paths = EnvConfig.get_paths()

        self.codenet_tar_path = os.path.join(self.paths['input_dir'], codenet_tar_path)
        self.metadata_tar_path = os.path.join(self.paths['input_dir'], metadata_tar_path)

        os.makedirs(self.paths['temp_dir'], exist_ok=True)
        os.makedirs(self.paths['output_dir'], exist_ok=True)

        logger.info(f"Using temporary directory: {self.paths['temp_dir']}")
        logger.info(f"Using output directory: {self.paths['output_dir']}")

    def extract_problem_list(self) -> pd.DataFrame:
        """Extract and read the problem_list.csv from metadata archive"""
        with tarfile.open(self.metadata_tar_path, 'r') as tar:
            problem_list_file = tar.extractfile('Project_CodeNet/metadata/problem_list.csv')
            if problem_list_file is None:
                raise FileNotFoundError("problem_list.csv not found in metadata archive")
            return pd.read_csv(problem_list_file)

    def extract_problem_metadata(self, problem_id: str) -> pd.DataFrame:
        """Extract and read metadata for a specific problem"""
        with tarfile.open(self.metadata_tar_path, 'r') as tar:
            metadata_file = tar.extractfile(f'Project_CodeNet/metadata/{problem_id}.csv')
            if metadata_file is None:
                raise FileNotFoundError(f"Metadata for problem {problem_id} not found")
            return pd.read_csv(metadata_file)

    def get_valid_problems(self, min_submissions: int = 5) -> List[str]:
        """
        Find problems that have sufficient accepted submissions in both Java and Python

        Args:
            min_submissions: Minimum number of accepted submissions required in each language
        Returns:
            List of problem IDs that meet the criteria
        """
        logger.info("Reading problem list...")
        problems = self.extract_problem_list()
        valid_problems = []
        total = len(problems)

        for idx, problem in problems.iterrows():
            problem_id = problem['id']
            try:
                if idx % 100 == 0:
                    logger.info(f"Processing problem {idx}/{total}")

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

        logger.info(f"Found {len(valid_problems)} valid problems")
        return valid_problems

    def extract_code_pair(self, problem_id: str, submission_id: str, language: str) -> str:
        """Extract source code for a specific submission"""
        with tarfile.open(self.codenet_tar_path, 'r') as tar:
            ext = 'py' if language == 'Python' else 'java'
            file_path = f'Project_CodeNet/data/{problem_id}/{language}/{submission_id}.{ext}'
            try:
                code_file = tar.extractfile(file_path)
                if code_file is None:
                    raise FileNotFoundError(f"Source file not found: {file_path}")
                return code_file.read().decode('utf-8', errors='replace')
            except Exception as e:
                logger.error(f"Error extracting {file_path}: {str(e)}")
                return None

    def create_dataset(self, output_dir: str, num_pairs: int = 10000, pairs_per_problem: int = 5):
        """
        Create the final dataset of Java-Python pairs

        Args:
            output_dir: Directory to save the extracted pairs
            num_pairs: Total number of pairs to extract
            pairs_per_problem: Maximum number of pairs to extract per problem
        """
        os.makedirs(output_dir, exist_ok=True)
        valid_problems = self.get_valid_problems()

        pairs_extracted = 0
        dataset = []

        for problem_id in valid_problems:
            if pairs_extracted >= num_pairs:
                break

            try:
                metadata = self.extract_problem_metadata(problem_id)
                accepted = metadata[metadata['status'] == 'Accepted']

                # Get submissions for both languages
                java_submissions = accepted[accepted['language'] == 'Java'].sort_values('cpu_time')[
                    'submission_id'].tolist()
                python_submissions = accepted[accepted['language'] == 'Python'].sort_values('cpu_time')[
                    'submission_id'].tolist()

                # Extract up to pairs_per_problem pairs
                for i in range(min(pairs_per_problem, len(java_submissions), len(python_submissions))):
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

                        if pairs_extracted % 100 == 0:
                            logger.info(f"Extracted {pairs_extracted} pairs...")
                            # Save intermediate results
                            intermediate_path = os.path.join(output_dir,
                                                             f'java_python_pairs_intermediate_{pairs_extracted}.json')
                            with open(intermediate_path, 'w') as f:
                                json.dump(dataset, f, indent=2)

            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                continue

        # Save the final dataset
        output_path = os.path.join(output_dir, 'java_python_pairs_final.json')
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Successfully extracted {len(dataset)} pairs to {output_path}")


def main():
    if DISABLE_PARAMS:
        # Direct execution with Kaggle configuration
        extractor = CodeNetExtractor(
            PARAMS_CONFIG['codenet_tar'],
            PARAMS_CONFIG['metadata_tar']
        )
        extractor.create_dataset(
            output_dir='extracted_pairs',
            num_pairs=PARAMS_CONFIG['num_pairs'],
            pairs_per_problem=PARAMS_CONFIG['pairs_per_problem']
        )
    else:
        # Local execution with command line arguments
        parser = argparse.ArgumentParser(description='Extract Java-Python pairs from CodeNet dataset')
        parser.add_argument('--codenet-tar', required=True, help='Path to Project_CodeNet.tar')
        parser.add_argument('--metadata-tar', required=True, help='Path to Project_CodeNet_metadata.tar')
        parser.add_argument('--output-dir', required=True, help='Output directory for extracted pairs')
        parser.add_argument('--num-pairs', type=int, default=10000, help='Number of pairs to extract')
        parser.add_argument('--pairs-per-problem', type=int, default=5, help='Maximum pairs per problem')

        args = parser.parse_args()
        extractor = CodeNetExtractor(args.codenet_tar, args.metadata_tar)
        extractor.create_dataset(args.output_dir, args.num_pairs, args.pairs_per_problem)


if __name__ == "__main__":
    main()