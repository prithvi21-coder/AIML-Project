import unittest

from project import DEMO_DOCUMENTS, TfidfRetriever, chunk_documents, local_answer


class UploadedDocumentAnswerTests(unittest.TestCase):
    def test_demo_guidance_is_about_lpu_and_uses_official_portal(self) -> None:
        self.assertEqual(len(DEMO_DOCUMENTS), 4)
        for document in DEMO_DOCUMENTS.values():
            self.assertIn("Lovely Professional University (LPU)", document)
            self.assertIn("https://admission.lpu.in/", document)

    def setUp(self) -> None:
        self.retriever = TfidfRetriever(
            chunk_documents(
                {
                    "scholarship.txt": (
                        "The scholarship application deadline is March 15. "
                        "Applicants must submit the FAFSA form and two "
                        "recommendation letters."
                    )
                }
            )
        )

    def test_paraphrased_deadline_question_returns_the_deadline(self) -> None:
        question = "What date must I apply by?"
        results = self.retriever.search(question)

        answer = local_answer(question, results)

        self.assertIn("deadline is March 15", answer)
        self.assertNotIn("recommendation letters", answer)

    def test_matching_deadline_terms_retrieve_uploaded_document(self) -> None:
        results = self.retriever.search("When is the scholarship deadline?")

        self.assertTrue(results)
        self.assertEqual(results[0][0].source, "scholarship.txt")


if __name__ == "__main__":
    unittest.main()
