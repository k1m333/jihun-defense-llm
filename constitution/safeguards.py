import os
from datetime import datetime

class ConstitutionalSafeguards:
    def __init__(self, rules_file="constitution/rules.txt"):
        self.forbidden_phrases = self._load_rules(rules_file)
        self.log_file = "constitution/audit.log"

    def _load_rules(self, path):
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            return [line.strip().lower() for line in f if line.strip()]

    def _log_refusal(self, prompt):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} | REFUSED: {prompt}\n")

    def check(self, prompt):
        """
        Returns (is_allowed, message)
        """
        prompt_lower = prompt.lower()
        for phrase in self.forbidden_phrases:
            if phrase in prompt_lower:
                self._log_refusal(prompt)
                return False, "I'm sorry, but I cannot assist with that request."
        return True, ""
