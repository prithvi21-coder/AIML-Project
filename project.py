"""Local-first RAG college admission guidance prototype.

Add institution-specific .txt or .md files to ``knowledge_base`` (or pass
``--knowledge-dir``) to ground answers in verified admissions information.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEMO_DOCUMENTS = {
    "demo_application_process.txt": """
College application process (general demonstration guidance)

Make a list of colleges and check each institution's official admissions
website for its current application steps, eligibility rules, required forms,
and deadlines. Requirements can differ by program and may change each year.
Submit the application and supporting documents through the channel specified
by the college. Keep copies of submitted materials and confirmation receipts.
Contact the admissions office when an official page does not answer a question.
""",
    "demo_documents.txt": """
Application documents (general demonstration guidance)

An application may request academic records, identification, test results,
recommendation letters, a personal statement, or other supporting materials.
The exact list depends on the college, program, and applicant category. Use the
current official checklist for the program you are applying to. Arrange for
records from other institutions to be sent in the format the college accepts.
Check that names and dates match across submitted documents.
""",
    "demo_financial_aid.txt": """
Financial aid (general demonstration guidance)

Financial aid and scholarship eligibility, application forms, and deadlines
vary between colleges and funding programs. Review the official financial aid
pages for each institution and check whether separate applications are needed.
Before sharing financial or identity documents, confirm the request and secure
submission channel with the institution. Contact the financial aid office
about eligibility questions or changes to your circumstances.
""",
    "demo_after_applying.txt": """
After applying (general demonstration guidance)

After submitting an application, check the institution's applicant portal and
email for status updates and requests for additional information. Respond by
the stated deadline and keep a copy of each communication. If a decision or
document is missing, contact the admissions office using contact details from
the institution's official website. Confirm enrollment and housing steps
directly with the college if you accept an offer.
""",
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "must", "need", "needed", "needs", "please", "should", "tell", "that",
    "the", "this", "to", "what", "when", "where", "which", "who", "with",
    "would", "you",
}

TERM_ALIASES = {
    "apply": {
        "applies", "applied", "applying", "application", "applications",
    },
    "deadline": {"deadlines", "date", "dates", "due"},
}
CANONICAL_TERMS = {
    alias: canonical
    for canonical, aliases in TERM_ALIASES.items()
    for alias in aliases | {canonical}
}


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


def tokenize(text: str) -> list[str]:
    return [
        CANONICAL_TERMS.get(word, word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    ]


def load_documents(directory: Path) -> tuple[dict[str, str], bool]:
    documents: dict[str, str] = {}
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                try:
                    content = path.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError) as error:
                    raise RuntimeError(f"Could not read {path}: {error}") from error
                if content:
                    documents[str(path.relative_to(directory))] = content
    if documents:
        return documents, False
    return DEMO_DOCUMENTS.copy(), True


def chunk_documents(documents: dict[str, str], chunk_size: int = 180) -> list[Passage]:
    passages: list[Passage] = []
    for source, content in documents.items():
        words = content.split()
        for start in range(0, len(words), chunk_size - 30):
            chunk = " ".join(words[start:start + chunk_size]).strip()
            if chunk:
                passages.append(Passage(source=source, text=chunk))
    return passages


class TfidfRetriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.term_counts = [Counter(tokenize(item.text)) for item in passages]
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        self.inverse_document_frequency = {
            term: math.log((len(passages) + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._weight(counts) for counts in self.term_counts]

    def _weight(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values())
        if not total:
            return {}
        return {
            term: (frequency / total) * self.inverse_document_frequency[term]
            for term, frequency in counts.items()
            if term in self.inverse_document_frequency
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        dot_product = sum(value * right.get(term, 0.0) for term, value in left.items())
        return dot_product / (left_norm * right_norm)

    def search(self, question: str, top_k: int = 3) -> list[tuple[Passage, float]]:
        query_counts = Counter(tokenize(question))
        query = self._weight(query_counts)
        if not query:
            return []
        ranked = [
            (passage, self._cosine(query, vector))
            for passage, vector in zip(self.passages, self.vectors)
        ]
        return sorted(
            (item for item in ranked if item[1] > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]


def local_answer(question: str, results: list[tuple[Passage, float]]) -> str:
    question_terms = set(tokenize(question))
    candidates: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for passage, _score in results:
        for sentence in re.split(r"(?<=[.!?])\s+", passage.text):
            clean_sentence = sentence.strip()
            key = clean_sentence.casefold()
            if not clean_sentence or key in seen:
                continue
            overlap = len(question_terms.intersection(tokenize(clean_sentence)))
            if overlap:
                seen.add(key)
                candidates.append((overlap, clean_sentence, passage.source))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return (
            "I couldn't find relevant information in the available guidance. "
            "Check the college's official admissions website or contact its "
            "admissions office."
        )
    return "\n".join(
        f"- {sentence} [{source}]"
        for _overlap, sentence, source in candidates[:4]
    )


def llm_answer(
    question: str,
    results: list[tuple[Passage, float]],
    api_key: str,
    model: str,
    base_url: str,
) -> str:
    context = "\n\n".join(
        f"[{index}] Source: {passage.source}\n{passage.text}"
        for index, (passage, _score) in enumerate(results, start=1)
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a college admissions guidance assistant. Answer "
                    "using only the supplied context. Do not invent institution-"
                    "specific requirements or dates. If the context is "
                    "insufficient, say so and direct the user to the college's "
                    "official admissions office. Cite claims with [1], [2], "
                    "etc., matching the supplied source numbers."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}",
            },
        ],
        "temperature": 0.2,
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed ({error.code}): {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the LLM service: {error.reason}") from error
    except (TimeoutError, json.JSONDecodeError, UnicodeError) as error:
        raise RuntimeError(f"Invalid or timed-out LLM response: {error}") from error

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("The LLM service returned an unexpected response.") from error
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The LLM service returned an empty answer.")
    return answer.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask college admission questions using retrieved local guidance."
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "knowledge_base",
        help="Folder containing UTF-8 .txt or .md guidance files.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Send the question and retrieved passages to an OpenAI-compatible API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        return 2
    if args.use_llm and not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY before using --use-llm.", file=sys.stderr)
        return 2

    try:
        documents, using_demo = load_documents(args.knowledge_dir)
        retriever = TfidfRetriever(chunk_documents(documents))
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if using_demo:
        print(
            "Using general demo guidance. Add verified college information "
            f"as .txt/.md files in {args.knowledge_dir} for institution-specific answers."
        )
    else:
        print(f"Loaded {len(documents)} guidance document(s) from {args.knowledge_dir}.")
    if args.use_llm:
        print("LLM mode is on: questions and retrieved passages will be sent to the configured API.")
    print("Ask an admissions question, or type 'quit' to exit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            return 0
        if not question:
            continue

        results = retriever.search(question, top_k=args.top_k)
        if not results:
            print(
                "\nGuidance: I couldn't find relevant information in the available "
                "guidance. Check the college's official admissions website or "
                "contact its admissions office."
            )
            continue
        try:
            if args.use_llm:
                answer = llm_answer(
                    question=question,
                    results=results,
                    api_key=os.environ["OPENAI_API_KEY"],
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    base_url=os.environ.get(
                        "OPENAI_BASE_URL", "https://api.openai.com/v1"
                    ),
                )
            else:
                answer = local_answer(question, results)
        except RuntimeError as error:
            print(f"\nError: {error}", file=sys.stderr)
            continue

        print(f"\nGuidance:\n{answer}")
        print("\nRetrieved sources:")
        for index, (passage, score) in enumerate(results, start=1):
            print(f"  [{index}] {passage.source} (similarity {score:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())