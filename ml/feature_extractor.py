"""
Feature Extraction, Document Parsing & Skill Gap Analysis Engine for AI Resume Analysis.

Capabilities:
1. Multi-Format Text Extraction: Robust parsing of .pdf, .docx, and .txt resumes with
   scanned document detection, OCR recommendations, and stream fallbacks.
2. Skill Extraction: Comprehensive regex pattern matching with lookaheads for tricky tokens
   (C++, C#, C, Go, R, Node.js, etc.) across 50+ industry technical competencies.
3. Skill Gap Analysis: Compares candidate skills against job requirements, identifies
   matched skills, detects missing skill gaps, and generates targeted learning recommendations.
4. TF-IDF Semantic Feature Extractor: Scikit-learn based vectorization and cosine similarity.
"""

import os
import re
import logging
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Multi-Format Document Parsing (.pdf, .docx, .txt)
# =====================================================================

def extract_text_from_pdf(file_obj_or_path):
    """
    Extracts plain text from a PDF document using pypdf.
    Detects scanned/image-based documents that yield zero or negligible text.

    Args:
        file_obj_or_path: Path string, file-like object, or BytesIO.

    Returns:
        tuple: (extracted_text: str, warning_or_error: str or None)
    """
    text = ""
    try:
        from pypdf import PdfReader

        if hasattr(file_obj_or_path, 'read'):
            reader = PdfReader(file_obj_or_path)
            if hasattr(file_obj_or_path, 'seek'):
                file_obj_or_path.seek(0)
        else:
            reader = PdfReader(str(file_obj_or_path))

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as exc:
        logger.error("Error reading PDF: %s", exc)
        return "", f"Error reading PDF: {str(exc)}"

    clean_text = text.strip()
    if not clean_text:
        return "", "Notice: No readable text extracted. This appears to be a scanned or image-based PDF. Scanned documents require an OCR engine (e.g. Tesseract OCR) to extract text."

    return clean_text, None


def extract_text_from_docx(file_obj_or_path):
    """
    Extracts text from a Word DOCX document paragraph by paragraph,
    including table cells.

    Args:
        file_obj_or_path: Path string, file-like object, or BytesIO.

    Returns:
        tuple: (extracted_text: str, warning_or_error: str or None)
    """
    text_parts = []
    try:
        from docx import Document

        if hasattr(file_obj_or_path, 'read'):
            doc = Document(file_obj_or_path)
            if hasattr(file_obj_or_path, 'seek'):
                file_obj_or_path.seek(0)
        else:
            doc = Document(str(file_obj_or_path))

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))

    except Exception as exc:
        logger.error("Error reading DOCX: %s", exc)
        return "", f"Error reading DOCX file: {str(exc)}"

    extracted_text = "\n".join(text_parts).strip()
    if not extracted_text:
        return "", "Notice: DOCX file is empty or contains only non-text media."

    return extracted_text, None


def extract_text_from_txt(file_obj_or_path):
    """
    Extracts text from a plain TXT file trying UTF-8, Latin-1, and CP1252 encodings.

    Args:
        file_obj_or_path: Path string, file-like object, or BytesIO.

    Returns:
        tuple: (extracted_text: str, warning_or_error: str or None)
    """
    try:
        if hasattr(file_obj_or_path, 'read'):
            raw_bytes = file_obj_or_path.read()
            if hasattr(file_obj_or_path, 'seek'):
                file_obj_or_path.seek(0)
        else:
            with open(file_obj_or_path, 'rb') as f:
                raw_bytes = f.read()

        if isinstance(raw_bytes, str):
            return raw_bytes.strip(), None

        # Try common encodings in order
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return raw_bytes.decode(enc).strip(), None
            except UnicodeDecodeError:
                continue

        return raw_bytes.decode('utf-8', errors='ignore').strip(), None

    except Exception as exc:
        logger.error("Error reading TXT: %s", exc)
        return "", f"Error reading TXT file: {str(exc)}"


def extract_resume_text(uploaded_file_or_path):
    """
    Main extraction dispatcher.
    Accepts an uploaded file object or file path and detects file type by extension.

    Returns:
        tuple: (extracted_text: str, warning_or_error: str or None)
    """
    filename = (
        uploaded_file_or_path.name
        if hasattr(uploaded_file_or_path, 'name')
        else str(uploaded_file_or_path)
    )
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(uploaded_file_or_path)
    elif ext == '.docx':
        return extract_text_from_docx(uploaded_file_or_path)
    elif ext == '.txt':
        return extract_text_from_txt(uploaded_file_or_path)
    else:
        return "", f"Unsupported file format '{ext}'. Only PDF (.pdf), Word (.docx), and Plain Text (.txt) files are supported."


# =====================================================================
# 2. Skill Extraction Patterns & Categorization
# =====================================================================

SKILL_PATTERNS = {
    # Programming Languages
    'Python': r'\bpython(?:3)?\b',
    'Java': r'\bjava\b(?!script)',
    'C++': r'(?<![a-zA-Z0-9#+])c\+\+(?![#+a-zA-Z])',
    'C#': r'(?<![a-zA-Z0-9#+])c#(?![#+a-zA-Z])|\bc-sharp\b|\bc\s*sharp\b',
    'C': (
        r'\bc\s+(?:programming|language|developer|compiler)\b|'
        r'\b(?:ansi\s+)?c\s+(?:compiler|code)\b|'
        r'\bembedded\s+c\b|'
        r'\bc\s*[/]\s*c\+\+|\bc\+\+\s*[/]\s*c\b|'
        r'\bc\s+(?:and|&)\s+c\+\+(?![a-zA-Z0-9#+])|\bc\+\+\s+(?:and|&)\s+c\b|'
        r'\b(?:python|java|c\+\+|sql|javascript|typescript|c#|rust|go|html|css)\s*,\s*c\s*[,.\n]|'
        r'\bc\s*,\s*(?:python|java|c\+\+|sql|rust|go|javascript|c#)\b|'
        r'\b(?:languages?|programming(?:\s+languages?)?)\s*:[^.\n]*?\bc\b'
    ),
    'JavaScript': r'\bjavascript\b|\bjs\b(?!on)',
    'TypeScript': r'\btypescript\b|\bts\b',
    'PHP': r'\bphp(?:7|8)?\b',
    'Ruby': r'\bruby\b',
    'Go': (
        r'\bgolang\b|'
        r'\bgo\s+(?:programming|language|developer|dev)\b|'
        r'\b(?:python|java|c\+\+|rust|docker|kubernetes|aws|sql)\s*(?:[,/]|and|&)\s*go\b|'
        r'\bgo\s*(?:[,/]|and|&)\s*(?:ruby|python|rust|docker|kubernetes|sql|c\+\+)\b|'
        r'\b(?:languages?|technologies)\s*:[^.\n]*?\bgo\b'
    ),
    'Rust': r'\brust\b',
    'Swift': r'\bswift\b',
    'Kotlin': r'\bkotlin\b',
    'R': (
        r'\br\s+(?:programming|language)\b|\br\s*studio\b|'
        r'\b(?:python|sql|julia|sas|spss|matlab)\s*(?:[,/]|and|&)\s*r\b|'
        r'\br\s*(?:[,/]|and|&)\s*(?:python|sql|julia|sas|spss|matlab)\b|'
        r'\b(?:languages?|statistical\s+tools?)\s*:[^.\n]*?\br\b'
    ),

    # Web & Frameworks
    'Django': r'\bdjango\b',
    'Flask': r'\bflask\b',
    'FastAPI': r'\bfastapi\b',
    'HTML': r'\bhtml(?:5)?\b',
    'CSS': r'\bcss(?:3)?\b',
    'Bootstrap': r'\bbootstrap(?:5)?\b',
    'Tailwind CSS': r'\btailwind(?:\s*css)?\b',
    'React': r'\breact(?:\.js|js)?\b',
    'Node.js': r'\bnode(?:\.js|js)?\b',
    'Express.js': r'\bexpress(?:\.js|js)?\b',
    'Angular': r'\bangular(?:\.js|js)?\b',
    'Vue.js': r'\bvue(?:\.js|js)?\b',
    'Spring Boot': r'\bspring\s*boot\b|\bspring\s+framework\b',

    # Databases & Storage
    'SQL': r'(?<!my)(?<!postgre)\bsql\b(?!ite)',
    'MySQL': r'\bmysql\b',
    'PostgreSQL': r'\bpostgres(?:ql)?\b',
    'SQLite': r'\bsqlite(?:3)?\b',
    'MongoDB': r'\bmongodb\b|\bmongo\b',
    'Redis': r'\bredis\b',
    'Oracle': r'\boracle\s*(?:db|database)?\b',

    # Data Science, ML & AI
    'Machine Learning': (
        r'\bmachine\s*learning\b|'
        r'\bml\s+(?:model|algorithm|pipeline|engineer|project|system|technique|method)s?\b|'
        r'\b(?:ai|nlp|dl)\s*(?:[,/]|&|and)\s*ml\b|'
        r'\bml\s*(?:[,/]|&|and)\s*(?:ai|nlp|dl)\b'
    ),
    'Deep Learning': (
        r'\bdeep\s*learning\b|'
        r'\bdl\s+(?:model|network|algorithm|engineer|system|framework)s?\b|'
        r'\b(?:ai|nlp|ml)\s*(?:[,/]|&|and)\s*dl\b|'
        r'\bdl\s*(?:[,/]|&|and)\s*(?:ai|nlp|ml)\b'
    ),
    'Natural Language Processing': r'\bnatural\s*language\s*processing\b|\bnlp\b',
    'Data Science': r'\bdata\s*science\b',
    'Pandas': r'\bpandas\b',
    'NumPy': r'\bnumpy\b',
    'Scikit-Learn': r'\bscikit-learn\b|\bsklearn\b',
    'TensorFlow': r'\btensorflow\b|\btf\b',
    'PyTorch': r'\bpytorch\b',
    'Keras': r'\bkeras\b',
    'OpenCV': r'\bopencv\b',
    'Data Analysis': r'\bdata\s*analysis\b',
    'Tableau': r'\btableau\b',
    'Power BI': r'\bpower\s*bi\b',
    'Big Data': r'\bbig\s*data\b',

    # Cloud, DevOps & Tools
    'Git': r'\bgit\b(?!hub|lab)',
    'GitHub': r'\bgithub\b',
    'GitLab': r'\bgitlab\b',
    'Docker': r'\bdocker\b',
    'Kubernetes': r'\bkubernetes\b|\bk8s\b',
    'AWS': r'\baws\b|\bamazon\s*web\s*services\b',
    'Azure': r'\bazure\b|\bmicrosoft\s*azure\b',
    'Google Cloud': r'\bgcp\b|\bgoogle\s*cloud\b',
    'Linux': r'\blinux\b|\bubuntu\b|\bcentos\b',
    'REST API': r'\brest(?:ful)?\s*api(?:s)?\b',
    'GraphQL': r'\bgraphql\b',
    'CI/CD': r'\bci\/cd\b|\bcontinuous\s*integration\b',
    'Postman': r'\bpostman\b',
    'Jira': r'\bjira\b',
    'Agile': r'\bagile\b|\bscrum\b',
}

SKILL_CATEGORIES = {
    'Programming Languages': ['Python', 'Java', 'C++', 'C#', 'C', 'JavaScript', 'TypeScript', 'PHP', 'Ruby', 'Go', 'Rust', 'Swift', 'Kotlin', 'R'],
    'Web & Frameworks': ['Django', 'Flask', 'FastAPI', 'HTML', 'CSS', 'Bootstrap', 'Tailwind CSS', 'React', 'Node.js', 'Express.js', 'Angular', 'Vue.js', 'Spring Boot'],
    'Databases': ['SQL', 'MySQL', 'PostgreSQL', 'SQLite', 'MongoDB', 'Redis', 'Oracle'],
    'AI, ML & Data Science': ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Data Science', 'Pandas', 'NumPy', 'Scikit-Learn', 'TensorFlow', 'PyTorch', 'Keras', 'OpenCV', 'Data Analysis', 'Tableau', 'Power BI', 'Big Data'],
    'DevOps, Cloud & Tools': ['Git', 'GitHub', 'GitLab', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'Google Cloud', 'Linux', 'REST API', 'GraphQL', 'CI/CD', 'Postman', 'Jira', 'Agile']
}

# Targeted learning recommendations for missing skills
SKILL_RECOMMENDATION_TEMPLATES = {
    'Docker': 'Gain practical experience with containerization: build multi-stage Dockerfiles and deploy microservices with Docker Compose.',
    'Kubernetes': 'Learn container orchestration: master Pods, Services, Deployments, and Helm charts for cluster workload management.',
    'AWS': 'Earn an AWS Certified Cloud Practitioner / Solutions Architect certification and build serverless apps with Lambda, S3, and RDS.',
    'Azure': 'Familiarize yourself with Microsoft Azure fundamentals (AZ-900), App Services, and Azure DevOps CI/CD pipelines.',
    'Google Cloud': 'Explore GCP core services (Compute Engine, BigQuery, Cloud Run, and GCS) and deployment architecture.',
    'CI/CD': 'Implement automated build, test, and deployment workflows using GitHub Actions or GitLab CI.',
    'Git': 'Practice version control best practices: branch protection, interactive rebasing, merge conflict resolution, and pull request workflows.',
    'SQL': 'Master complex SQL queries: window functions, CTEs, query optimization, indexing strategies, and database normalization.',
    'PostgreSQL': 'Work with PostgreSQL advanced features: JSONB querying, full-text search, pgvector, and query plan profiling.',
    'MongoDB': 'Explore document data modeling, aggregation pipelines, replica sets, and indexing in NoSQL MongoDB.',
    'Redis': 'Implement in-memory caching, session storage, and rate limiting using Redis key-value structures.',
    'REST API': 'Design and document RESTful APIs following OpenAPI/Swagger specifications with JWT authentication and rate limiting.',
    'GraphQL': 'Learn schema-first GraphQL design: queries, mutations, resolvers, and subscriptions with Apollo or Strawberry.',
    'Python': 'Deepen Python knowledge: async/await with asyncio, generators, type hints, decorators, and clean architectural design patterns.',
    'Django': 'Build robust web applications with Django ORM, authentication, Django REST Framework, and background Celery tasks.',
    'FastAPI': 'Develop high-performance asynchronous microservices using FastAPI, Pydantic validation, and OpenAPI documentation.',
    'React': 'Master modern React: functional components, React Hooks (useEffect, useMemo), state management, and Vite tooling.',
    'Node.js': 'Build event-driven asynchronous backend services with Node.js, Express, and npm package ecosystems.',
    'TypeScript': 'Adopt TypeScript for type-safe frontend or backend development: interfaces, generics, and strict compiler configs.',
    'Machine Learning': 'Implement end-to-end ML pipelines: feature engineering, cross-validation, hyperparameter tuning, and model evaluation metrics.',
    'Deep Learning': 'Build neural network architectures (CNNs, RNNs, Transformers) using PyTorch or TensorFlow for computer vision or NLP.',
    'Natural Language Processing': 'Explore modern NLP: text preprocessing, TF-IDF, Word2Vec, and fine-tuning Transformer models (Hugging Face BERT).',
    'Pandas': 'Practice data manipulation, aggregation, time series resample, and multi-index transformations with Pandas.',
    'NumPy': 'Strengthen vector and matrix mathematical operations, broadcasting, and numerical computing with NumPy.',
    'Scikit-Learn': 'Build reproducible scikit-learn pipelines with ColumnTransformer, StandardScaler, and estimators.',
    'TensorFlow': 'Develop and train neural networks using TensorFlow/Keras and deploy models via TensorFlow Serving.',
    'PyTorch': 'Build deep learning architectures using PyTorch tensors, autograd, custom Datasets, and DataLoader pipelines.',
    'Linux': 'Familiarize yourself with Linux CLI navigation, bash scripting, permission models, systemd services, and SSH tooling.',
}


def extract_skills_from_text(text):
    """
    Extracts all recognized technical skills from the provided text.

    Returns:
        dict: {
            'skills': list of recognized skill strings (alphabetical),
            'categorized': dict mapping category names to lists of skills,
            'total_count': total number of unique skills found
        }
    """
    if not text:
        return {'skills': [], 'categorized': {}, 'total_count': 0}

    lower_text = text.lower()
    extracted = set()

    for skill_name, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, lower_text):
            extracted.add(skill_name)

    skills_list = sorted(list(extracted))

    categorized = {}
    for cat_name, skill_members in SKILL_CATEGORIES.items():
        matched = [s for s in skills_list if s in skill_members]
        if matched:
            categorized[cat_name] = matched

    return {
        'skills': skills_list,
        'categorized': categorized,
        'total_count': len(skills_list)
    }


# =====================================================================
# 3. Skill Gap Analysis
# =====================================================================

def analyze_skill_gap(resume_input, job_description_text):
    """
    Performs a comprehensive Skill Gap Analysis between a candidate's resume
    and the employer's Job Description.

    Args:
        resume_input: Either raw resume text (str) or a pre-extracted list of skills (list/set).
        job_description_text: Raw target job description text (str).

    Returns:
        dict: {
            'job_skills': list of skills demanded by the JD,
            'resume_skills': list of skills in the resume,
            'matched_skills': list of overlapping skills (badges),
            'missing_skills': list of absent required skills,
            'match_percentage': float (0.0 to 100.0),
            'gap_severity': 'Low' | 'Moderate' | 'Critical',
            'recommendations': list of targeted improvement recommendations
        }
    """
    if not job_description_text or not job_description_text.strip():
        # No job description provided
        resume_skills = (
            list(resume_input)
            if isinstance(resume_input, (list, set))
            else extract_skills_from_text(str(resume_input))['skills']
        )
        return {
            'job_skills': [],
            'resume_skills': sorted(resume_skills),
            'matched_skills': [],
            'missing_skills': [],
            'match_percentage': 0.0,
            'gap_severity': 'N/A',
            'recommendations': ['Provide a target Job Description to receive tailored Skill Gap Analysis and recommendations.']
        }

    # Extract resume skills
    if isinstance(resume_input, (list, set)):
        resume_skills_set = set(resume_input)
    else:
        resume_skills_set = set(extract_skills_from_text(str(resume_input))['skills'])

    # Extract JD skills
    jd_extraction = extract_skills_from_text(job_description_text)
    jd_skills_set = set(jd_extraction['skills'])

    matched = sorted(list(resume_skills_set.intersection(jd_skills_set)))
    missing = sorted(list(jd_skills_set.difference(resume_skills_set)))

    if jd_skills_set:
        match_percentage = round((len(matched) / len(jd_skills_set)) * 100, 1)
    else:
        match_percentage = 0.0

    # Determine gap severity
    if match_percentage >= 75.0:
        gap_severity = 'Low Gap'
    elif match_percentage >= 45.0:
        gap_severity = 'Moderate Gap'
    else:
        gap_severity = 'Critical Gap'

    # Build recommendations for missing skills
    recommendations = []
    if missing:
        for skill in missing:
            custom_rec = SKILL_RECOMMENDATION_TEMPLATES.get(
                skill,
                f"Develop proficiency in {skill} through hands-on project work, tutorials, and targeted portfolio additions."
            )
            recommendations.append(f"{skill}: {custom_rec}")
    else:
        if jd_skills_set:
            recommendations.append(
                "Outstanding alignment! Your resume covers 100% of the recognized technical skills demanded in this job description."
            )
        else:
            recommendations.append(
                "No standard technical keywords were recognized in the job description. Verify whether the role focuses on non-technical competencies."
            )

    return {
        'job_skills': sorted(list(jd_skills_set)),
        'resume_skills': sorted(list(resume_skills_set)),
        'matched_skills': matched,
        'missing_skills': missing,
        'match_percentage': match_percentage,
        'gap_severity': gap_severity,
        'recommendations': recommendations,
        'total_job_skills': len(jd_skills_set),
        'total_matched': len(matched),
        'total_missing': len(missing)
    }


# Backwards compatibility alias for skill comparison
def compare_with_job_description(resume_skills, job_description_text, resume_text=""):
    """
    Backwards compatibility adapter for resumeapp services.
    """
    gap_result = analyze_skill_gap(resume_skills, job_description_text)

    # Optional TF-IDF semantic similarity
    semantic_similarity = 0.0
    if resume_text and job_description_text:
        try:
            extractor = ResumeFeatureExtractor()
            semantic_similarity = extractor.compute_similarity(resume_text, job_description_text)
        except Exception:
            semantic_similarity = 0.0

    return {
        'job_skills': gap_result['job_skills'],
        'matched_skills': gap_result['matched_skills'],
        'missing_skills': gap_result['missing_skills'],
        'match_percentage': gap_result['match_percentage'],
        'semantic_similarity': semantic_similarity,
        'recommendations': gap_result['recommendations'],
        'gap_severity': gap_result['gap_severity'],
    }


# =====================================================================
# 4. Scikit-Learn TF-IDF Feature Extractor
# =====================================================================

class ResumeFeatureExtractor:
    """
    Extracts TF-IDF features and measures keyword similarity
    between resumes and job descriptions using scikit-learn.
    """
    def __init__(self, max_features=1000, stop_words='english'):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=(1, 2)
        )
        self.is_fitted = False

    def fit_transform(self, text_corpus):
        """Fits vectorizer to a corpus of resumes and transforms it to TF-IDF matrix."""
        self.is_fitted = True
        return self.vectorizer.fit_transform(text_corpus)

    def compute_similarity(self, resume_text, job_description_text):
        """
        Calculates cosine similarity between a resume and job description.
        Returns: float percentage between 0.0 and 100.0.
        """
        if not resume_text or not job_description_text:
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_text, job_description_text])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(round(max(0.0, min(1.0, sim)) * 100, 1))
        except Exception:
            return 0.0
