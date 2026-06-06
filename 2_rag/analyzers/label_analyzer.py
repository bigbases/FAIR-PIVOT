import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import Counter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseAnalyzer
from core.types import Perspective


class LabelAnalyzer(BaseAnalyzer):
    """Analyzes political labels using major voting across multiple GPT-4.1 persona annotations"""

    def __init__(self):
        self.gpt_cols = [
            'gpt-4.1_opp_left',
            'gpt-4.1_opp_right',
            'gpt-4.1_sup_left',
            'gpt-4.1_sup_right'
        ]

    def analyze(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze political labels for each row using major voting

        For each sample:
        1. Collect Political labels/scores from 4 personas
        2. Apply majority voting for Political (Left/Center/Right)
        3. Use average score to resolve ties (-0.2 to 0.2 -> Center)
        4. Expose the result as both 'political_major' and 'main_perspective'
        """
        results: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            row_result: Dict[str, Any] = {'id': row['id']}

            # Collect annotations from all 4 personas
            annotations = []
            for col in self.gpt_cols:
                pol_label, pol_score = self._extract_label_and_score(row[col])
                annotations.append((pol_label, pol_score))

            # Determine major political label
            major_political = self._determine_major_political(annotations)

            # Store results
            row_result['political_major'] = major_political
            row_result['main_perspective'] = major_political

            # Store individual annotations for reference
            for i, (pol_label, pol_score) in enumerate(annotations):
                persona = self.gpt_cols[i].replace('gpt-4.1_', '')
                row_result[f'{persona}_political'] = pol_label
                row_result[f'{persona}_political_score'] = pol_score

            results.append(row_result)

        return results

    def _extract_label_and_score(self, json_str: str) -> Tuple[str, float]:
        """
        Extract political label and score from GPT annotation

        Returns:
            (political_label, political_score)
        """
        try:
            data = json.loads(json_str)
            political_label = data.get('Political', {}).get('label', 'Undecided')
            political_score = data.get('Political', {}).get('score', 0.0)
            return political_label, political_score
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 'Undecided', 0.0

    def _determine_major_political(self, annotations: List[Tuple[str, float]]) -> str:
        """
        Determine major political label using majority voting

        Args:
            annotations: List of (political_label, political_score)

        Returns:
            major_political
        """
        # Separate votes and scores
        political_votes = [ann[0] for ann in annotations]
        political_scores = [ann[1] for ann in annotations]

        # Calculate average score for tie-breaking
        avg_political_score = sum(political_scores) / len(political_scores) if political_scores else 0.0

        # Major voting for political
        political_counter = Counter(political_votes)
        political_most_common = political_counter.most_common(2)

        if len(political_most_common) == 1:
            # Only one label exists
            major_political = political_most_common[0][0]
        elif political_most_common[0][1] > political_most_common[1][1]:
            # Clear majority (top vote count > second vote count)
            major_political = political_most_common[0][0]
        else:
            # Tie: use average score to decide
            if -0.2 <= avg_political_score <= 0.2:
                major_political = 'Center'
            elif avg_political_score < 0:
                major_political = 'Left'
            else:
                major_political = 'Right'

        return major_political
