"""
Automated Unit Test Suite for Machine Learning & Document Extraction Pipeline.

Covers:
1. Multi-format text extraction (.pdf, .docx, .txt, unsupported formats, scanned warnings).
2. Skill extraction, regex token lookaheads (C++, C#, Go, R, C), and anti-false-positive rules.
3. Skill Gap Analysis (matched skills, missing skills, gap percentages, and targeted recommendations).
4. Local BERT SentenceTransformer Singleton loader, thread-safety, and cosine similarity scoring.
5. Scikit-learn career role prediction and class probability distributions.
"""

import os
from io import BytesIO
from unittest.mock import patch, MagicMock
from django.test import SimpleTestCase, TestCase

from ml.classifier import (
    BertModelSingleton,
    get_bert_model,
    calculate_match_score,
    predict_role,
    classify_resume_text,
    get_classifier_model,
)
from ml.feature_extractor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_resume_text,
    extract_skills_from_text,
    analyze_skill_gap,
    ResumeFeatureExtractor,
)


class DocumentExtractionTests(SimpleTestCase):
    """Tests multi-format resume text extraction (.pdf, .docx, .txt)."""

    def test_txt_extraction_utf8(self):
        """Verify plain text extraction from UTF-8 byte streams."""
        content = "John Doe\nPython Developer\nSkills: Python, Django, SQL"
        stream = BytesIO(content.encode('utf-8'))
        stream.name = "resume.txt"
        text, error = extract_text_from_txt(stream)
        self.assertIsNone(error)
        self.assertIn("Python Developer", text)
        self.assertIn("SQL", text)

    def test_txt_extraction_latin1(self):
        """Verify plain text extraction fallback with Latin-1 encoding."""
        content = "Resume with special accents: café, résumé, Zürich"
        stream = BytesIO(content.encode('latin-1'))
        stream.name = "resume.txt"
        text, error = extract_text_from_txt(stream)
        self.assertIsNone(error)
        self.assertIn("Zürich", text)

    def test_docx_extraction(self):
        """Verify DOCX text extraction including paragraphs and table contents."""
        from docx import Document

        doc = Document()
        doc.add_heading("Jane Doe - Machine Learning Engineer", level=1)
        doc.add_paragraph("Proficient in Python, TensorFlow, and PyTorch.")

        table = doc.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Skill"
        table.rows[0].cells[1].text = "Proficiency"
        table.rows[1].cells[0].text = "Machine Learning"
        table.rows[1].cells[1].text = "Expert"

        stream = BytesIO()
        doc.save(stream)
        stream.seek(0)
        stream.name = "resume.docx"

        text, error = extract_text_from_docx(stream)
        self.assertIsNone(error)
        self.assertIn("Machine Learning Engineer", text)
        self.assertIn("TensorFlow", text)
        self.assertIn("Machine Learning", text)

    def test_pdf_extraction_sample_file(self):
        """Verify text extraction from a real sample PDF in media/resumes/."""
        sample_path = os.path.join(
            os.path.dirname(__file__), '..', 'media', 'resumes', 'Ankush_Baghel_Resume-3_1.pdf'
        )
        if os.path.isfile(sample_path):
            text, error = extract_text_from_pdf(sample_path)
            self.assertIsNone(error)
            self.assertTrue(len(text) > 50)
            self.assertTrue(any(k in text.lower() for k in ['education', 'skills', 'experience', 'projects']))

    def test_pdf_extraction_in_memory(self):
        """Verify PDF text extraction using pypdf writer."""
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        stream = BytesIO()
        writer.write(stream)
        stream.seek(0)
        stream.name = "blank.pdf"

        text, warning = extract_text_from_pdf(stream)
        # Blank PDF should yield warning about scanned/image document
        self.assertEqual(text, "")
        self.assertIsNotNone(warning)
        self.assertIn("scanned or image-based", warning.lower())

    def test_extract_resume_text_dispatcher(self):
        """Verify dispatcher correctly routes based on file extension."""
        txt_stream = BytesIO(b"Candidate Name\nPython Engineer")
        txt_stream.name = "candidate.txt"
        text, err = extract_resume_text(txt_stream)
        self.assertIsNone(err)
        self.assertIn("Python Engineer", text)

        invalid_stream = BytesIO(b"\x89PNG\r\n\x1a\n")
        invalid_stream.name = "photo.png"
        text, err = extract_resume_text(invalid_stream)
        self.assertEqual(text, "")
        self.assertIn("unsupported file format", err.lower())


class SkillExtractionAndGapAnalysisTests(SimpleTestCase):
    """Tests technical skill extraction and Job Description gap analysis."""

    def test_extract_skills_from_text(self):
        """Verify detection of programming languages, web frameworks, and cloud tools."""
        sample_text = (
            "Full Stack Software Engineer experienced in Python, Django, React, "
            "PostgreSQL, Docker, AWS, and Git. Familiar with Machine Learning pipelines."
        )
        result = extract_skills_from_text(sample_text)
        extracted = result['skills']

        self.assertIn('Python', extracted)
        self.assertIn('Django', extracted)
        self.assertIn('React', extracted)
        self.assertIn('PostgreSQL', extracted)
        self.assertIn('Docker', extracted)
        self.assertIn('AWS', extracted)
        self.assertIn('Git', extracted)
        self.assertIn('Machine Learning', extracted)
        self.assertGreaterEqual(result['total_count'], 7)

    def test_tricky_skill_boundary_cases(self):
        """Verify that tricky tokens (C++, C#, Go, R, C) match correctly without false positives."""
        text = "Proficient in C++, C#, and Python. Experience with Go programming and R studio."
        result = extract_skills_from_text(text)
        skills = result['skills']

        self.assertIn('C++', skills)
        self.assertIn('C#', skills)
        self.assertIn('Python', skills)
        self.assertIn('Go', skills)
        self.assertIn('R', skills)

        # Negative false-positive checks: regular prose should NOT trigger false language matches
        prose_text = "The doctor directed the committee to check the recipe."
        prose_result = extract_skills_from_text(prose_text)
        self.assertNotIn('C', prose_result['skills'])
        self.assertNotIn('R', prose_result['skills'])
        self.assertNotIn('Go', prose_result['skills'])

    def test_analyze_skill_gap_calculation(self):
        """Verify matched vs missing skills and percentage calculation."""
        resume_skills = ['Python', 'Django', 'SQL', 'Git']
        job_description = (
            "We are seeking a Backend Developer skilled in Python, Django, SQL, "
            "Docker, Kubernetes, and AWS."
        )

        gap_result = analyze_skill_gap(resume_skills, job_description)

        # Matched skills
        self.assertIn('Python', gap_result['matched_skills'])
        self.assertIn('Django', gap_result['matched_skills'])
        self.assertIn('SQL', gap_result['matched_skills'])

        # Missing skills
        self.assertIn('Docker', gap_result['missing_skills'])
        self.assertIn('Kubernetes', gap_result['missing_skills'])
        self.assertIn('AWS', gap_result['missing_skills'])

        # Total JD skills = 6, Matched = 3 -> 50.0%
        self.assertEqual(gap_result['match_percentage'], 50.0)
        self.assertEqual(gap_result['gap_severity'], 'Moderate Gap')

        # Targeted recommendations
        recommendations = " ".join(gap_result['recommendations'])
        self.assertIn('Docker', recommendations)
        self.assertIn('Kubernetes', recommendations)
        self.assertIn('AWS', recommendations)

    def test_analyze_skill_gap_full_match(self):
        """Verify 100% skill match with zero missing skills."""
        resume_skills = ['Python', 'FastAPI', 'Docker', 'PostgreSQL']
        job_description = "Requirements: Python, FastAPI, Docker, and PostgreSQL."

        gap_result = analyze_skill_gap(resume_skills, job_description)
        self.assertEqual(gap_result['match_percentage'], 100.0)
        self.assertEqual(len(gap_result['missing_skills']), 0)
        self.assertEqual(gap_result['gap_severity'], 'Low Gap')
        self.assertTrue(any("100%" in r or "Outstanding" in r for r in gap_result['recommendations']))

    def test_analyze_skill_gap_empty_inputs(self):
        """Verify graceful handling when job description is empty."""
        gap_result = analyze_skill_gap(['Python', 'Django'], "")
        self.assertEqual(gap_result['match_percentage'], 0.0)
        self.assertEqual(gap_result['matched_skills'], [])
        self.assertEqual(gap_result['missing_skills'], [])

    def test_resume_feature_extractor_tfidf_similarity(self):
        """Verify scikit-learn TF-IDF cosine similarity computation."""
        extractor = ResumeFeatureExtractor()
        resume_text = "Python developer with Django and SQL database experience."
        jd_text = "Python developer with Django and SQL database experience."
        sim_identical = extractor.compute_similarity(resume_text, jd_text)
        self.assertAlmostEqual(sim_identical, 100.0, delta=1.0)

        unrelated_text = "Baking fresh sourdough bread in a hot brick oven."
        sim_unrelated = extractor.compute_similarity(resume_text, unrelated_text)
        self.assertEqual(sim_unrelated, 0.0)


class BertCosineSimilarityAndSingletonTests(SimpleTestCase):
    """Tests local BERT SentenceTransformer Singleton loader and cosine similarity."""

    def test_singleton_loader_identity(self):
        """Verify that repeated requests return the exact same model instance (Singleton)."""
        instance1 = BertModelSingleton()
        instance2 = BertModelSingleton()
        self.assertIs(instance1, instance2)

        model1 = instance1.get_model()
        model2 = instance2.get_model()
        self.assertIsNotNone(model1)
        self.assertIs(model1, model2)

    def test_get_bert_model_helper(self):
        """Verify get_bert_model returns the singleton instance."""
        model = get_bert_model()
        self.assertIsNotNone(model)
        self.assertIs(model, BertModelSingleton().get_model())

    def test_bert_cosine_similarity_relevance(self):
        """Verify semantic similarity is high for matching topics and lower for unrelated topics."""
        singleton = BertModelSingleton()

        text_dev1 = "Experienced Python and Django backend web developer building REST APIs and microservices."
        text_dev2 = "Seeking a Python software engineer proficient in Django web frameworks and API design."
        text_unrelated = "Delicious Italian recipe for making homemade pasta with fresh basil, tomatoes, and garlic."

        sim_related = singleton.compute_similarity(text_dev1, text_dev2)
        sim_unrelated = singleton.compute_similarity(text_dev1, text_unrelated)

        self.assertIsNotNone(sim_related)
        self.assertIsNotNone(sim_unrelated)
        self.assertGreater(sim_related, 0.65)
        self.assertGreater(sim_related, sim_unrelated)

    def test_calculate_match_score(self):
        """Verify calculate_match_score outputs expected dictionary format and score bounds."""
        resume = "Machine Learning Engineer with deep learning, PyTorch, and NLP background."
        jd = "Hiring Machine Learning Engineer to train deep learning models and NLP Transformers."

        score_data = calculate_match_score(resume, jd)
        self.assertTrue(score_data['available'])
        self.assertIn('match_score', score_data)
        self.assertIn('cosine_similarity', score_data)
        self.assertIn('match_level', score_data)
        self.assertIn('model_name', score_data)
        self.assertGreater(score_data['match_score'], 50.0)
        self.assertLessEqual(score_data['match_score'], 100.0)

    def test_calculate_match_score_empty_inputs(self):
        """Verify calculate_match_score handles empty strings gracefully."""
        res = calculate_match_score("", "Some job description")
        self.assertFalse(res['available'])
        self.assertEqual(res['match_score'], 0.0)

        res2 = calculate_match_score("Some resume text", "")
        self.assertFalse(res2['available'])
        self.assertEqual(res2['match_score'], 0.0)


class ClassifierRolePredictionTests(SimpleTestCase):
    """Tests scikit-learn career role classification pipeline."""

    def test_predict_role_output_structure(self):
        """Verify predict_role returns prediction, confidence, and class probabilities."""
        text = (
            "Data Scientist skilled in Machine Learning, Deep Learning, Python, "
            "Pandas, Scikit-Learn, TensorFlow, statistical modeling, and data analysis."
        )
        prediction = predict_role(text)
        self.assertIsNotNone(prediction)
        self.assertIn('predicted_role', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('class_probabilities', prediction)
        self.assertIn('model_name', prediction)

        # Confidence should be a float between 0 and 100
        self.assertGreater(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 100.0)

        # Probabilities should contain multiple roles
        self.assertIsInstance(prediction['class_probabilities'], dict)
        self.assertGreater(len(prediction['class_probabilities']), 1)

    def test_classify_resume_text_backwards_compatibility(self):
        """Verify classify_resume_text acts as a drop-in alias for predict_role."""
        text = "Frontend Web Developer with HTML, CSS, JavaScript, React, and Bootstrap."
        result = classify_resume_text(text)
        self.assertIsNotNone(result)
        self.assertIn('predicted_role', result)

    def test_predict_role_empty_input(self):
        """Verify predict_role safely returns None for empty strings."""
        self.assertIsNone(predict_role(""))
        self.assertIsNone(predict_role("   "))
