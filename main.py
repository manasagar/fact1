import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
import re
from typing import List, Dict, Tuple, Any, Optional
import logging
from flask import Flask,request,jsonify
# Configure logging
pipe=None
document=""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app=Flask(__name__)
@app.route("/echo",methods=["POST"])
def echo():
    res=request.json
    res=res['text']
    global pipe
    results=pipe.process_document(document,res)
    print(pipe.format_results(results))
    return jsonify(pipe.format_results(results)),200

class ClaimExtractionVerificationPipeline:
    def __init__(self, use_gpu=False):
        """
        Initialize the pipeline with required models.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load NLI model for claim detection and verification
        logger.info("Loading NLI model...")
        self.nli_model_name = "facebook/bart-large-mnli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name).to(self.device)

        # Load sentence transformer for semantic search
        logger.info("Loading sentence embedding model...")
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2').to(self.device)

        # Load NER model
        logger.info("Loading NER pipeline...")
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=0 if self.device == "cuda" else -1)

        # Load spaCy for linguistic processing
        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")

        logger.info("All models loaded successfully!")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences

    def _preprocess_document(self, document: str) -> List[str]:
        """Process document into chunks for fact-checking."""
        # Split document into sentences
        sentences = self._split_into_sentences(document)

        # Create chunks of sentences for context
        chunks = []
        chunk_size = 3  # Number of sentences per chunk
        overlap = 1     # Overlap between chunks

        for i in range(0, len(sentences), chunk_size - overlap):
            if i + chunk_size <= len(sentences):
                chunk = ' '.join(sentences[i:i+chunk_size])
            else:
                chunk = ' '.join(sentences[i:])

            if len(chunk.split()) >= 5:  # Only keep chunks with at least 5 words
                chunks.append(chunk)

        return chunks

    def _is_claim(self, sentence: str) -> Tuple[bool, float]:
        """Determine if a sentence contains a claim."""
        # Skip very short sentences or questions
        if len(sentence.split()) < 3 or sentence.strip().endswith('?'):
            return False, 0.0

        # Template approach to check if sentence is a factual claim
        template = "This sentence states a factual claim."

        # Tokenize
        inputs = self.nli_tokenizer(sentence, template, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        # Check entailment score (MNLI labels: 0=contradiction, 1=neutral, 2=entailment)
        entailment_score = probabilities[2].item()
        is_claim = entailment_score > 0.6

        return is_claim, entailment_score

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        entities = self.ner_pipeline(text)

        # Group consecutive entity tokens
        grouped_entities = []
        current_entity = None

        for entity in entities:
            if current_entity is None:
                current_entity = {
                    "entity_group": entity["entity"],
                    "word": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": entity["score"]
                }
            elif (entity["entity"] == current_entity["entity_group"] and
                  entity["start"] == current_entity["end"]):
                # Extend the current entity
                current_entity["word"] += entity["word"].replace("##", "")
                current_entity["end"] = entity["end"]
                current_entity["score"] = (current_entity["score"] + entity["score"]) / 2
            else:
                # Start a new entity
                grouped_entities.append(current_entity)
                current_entity = {
                    "entity_group": entity["entity"],
                    "word": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": entity["score"]
                }

        if current_entity:
            grouped_entities.append(current_entity)

        return grouped_entities

    def _extract_claim_structure(self, sentence: str) -> Dict:
        """Extract subject-predicate-object structure from a claim."""
        doc = self.nlp(sentence)

        # Initialize structure components
        subject = ""
        predicate = ""
        obj = ""

        # Try to identify the main verb and its arguments
        for token in doc:
            if token.dep_ == "ROOT":
                predicate = token.lemma_

                # Find subject
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = ' '.join([w.text for w in child.subtree])
                        break

                # Find object
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj", "attr"):
                        obj = ' '.join([w.text for w in child.subtree])
                        break

        # Return structure information
        return {
            "subject": subject.strip(),
            "predicate": predicate.strip(),
            "object": obj.strip(),
            "structured": bool(subject and predicate)
        }

    def _find_relevant_evidence(self, claim: str, document_chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Find most relevant document chunks for a claim using semantic search."""
        # Get embeddings
        claim_embedding = self.sentence_model.encode(claim, convert_to_tensor=True)
        chunk_embeddings = self.sentence_model.encode(document_chunks, convert_to_tensor=True)

        # Calculate similarity
        similarities = util.cos_sim(claim_embedding, chunk_embeddings)[0]

        # Get top-k chunks
        top_indices = torch.argsort(similarities, descending=True)[:top_k].tolist()
        top_chunks = [(document_chunks[i], similarities[i].item()) for i in top_indices]

        return top_chunks

    def _verify_claim(self, claim: str, evidence: str) -> Dict:
        """Verify a claim against evidence using NLI."""
        # Tokenize
        inputs = self.nli_tokenizer(evidence, claim, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        # Get prediction class and confidence
        prediction_idx = torch.argmax(probabilities).item()

        # Map indices to labels
        idx_to_label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        nli_label = idx_to_label[prediction_idx]
        confidence = probabilities[prediction_idx].item()

        # Map NLI labels to our desired labels
        if nli_label == "entailment":
            verdict = "SUPPORTED"
        elif nli_label == "contradiction":
            verdict = "REFUTED"
        else:
            verdict = "NOT_ENOUGH_INFO"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "entailment_score": probabilities[2].item(),
            "contradiction_score": probabilities[0].item(),
            "neutral_score": probabilities[1].item()
        }

    def extract_claims(self, text: str, min_confidence: float = 0.6) -> List[Dict]:
        """Extract claims from text."""
        # Split text into sentences
        sentences = self._split_into_sentences(text)

        # Find claims
        claims = []
        for i, sentence in enumerate(sentences):
            is_claim, confidence = self._is_claim(sentence)

            if is_claim and confidence >= min_confidence:
                # Get entities and structure
                entities = self._extract_entities(sentence)
                structure = self._extract_claim_structure(sentence)

                # Add to claims list
                claims.append({
                    "text": sentence,
                    "confidence": confidence,
                    "position": i,
                    "entities": entities,
                    "structure": structure,
                })

        return claims

    def verify_claims(self, claims: List[Dict], document: str) -> List[Dict]:
        """Verify extracted claims against the document."""
        # Preprocess document into chunks
        document_chunks = self._preprocess_document(document)

        # Verify each claim
        verified_claims = []

        for claim in claims:
            # Find relevant evidence
            relevant_evidence = self._find_relevant_evidence(claim["text"], document_chunks)

            # Verify against each piece of evidence
            verification_results = []
            for evidence_text, relevance_score in relevant_evidence:
                verification = self._verify_claim(claim["text"], evidence_text)
                verification_results.append({
                    "evidence": evidence_text,
                    "relevance": relevance_score,
                    "verification": verification
                })

            # Determine final verdict using the most confident verification
            # Sort by confidence and entailment/contradiction scores
            verification_results.sort(key=lambda x:
                max(x["verification"]["entailment_score"],
                    x["verification"]["contradiction_score"]) * x["relevance"],
                reverse=True)

            # Select the most confident result that's not neutral
            best_verification = next((v for v in verification_results
                                    if v["verification"]["verdict"] != "NOT_ENOUGH_INFO" and
                                    v["verification"]["confidence"] > 0.7),
                                   verification_results[0] if verification_results else None)

            if best_verification:
                final_verdict = best_verification["verification"]["verdict"]
                confidence = best_verification["verification"]["confidence"]
                evidence = best_verification["evidence"]
            else:
                final_verdict = "NOT_ENOUGH_INFO"
                confidence = 0.5
                evidence = ""

            # Add verification information to the claim
            claim["verification"] = {
                "verdict": final_verdict,
                "confidence": confidence,
                "evidence": evidence,
                "all_results": verification_results
            }

            verified_claims.append(claim)

        return verified_claims

    def process_document(self, document: str, claims: Optional[List[str]] = None) -> Dict:
        """
        Process a document to extract and verify claims.

        Args:
            document: Text document to analyze
            claims: Optional list of specific claims to verify
                   (if None, claims will be extracted from the document)

        Returns:
            Dictionary with extracted claims and verification results
        """
        logger.info("Processing document...")

        # Extract claims if not provided
        if claims :
            logger.info("Extracting claims from document...")
            extracted_claims = self.extract_claims(claims)
            logger.info(f"Found {len(extracted_claims)} claims")
        else:
            logger.info(f"Using {len(claims)} provided claims")
            extracted_claims = [{"text": claim, "position": -1} for claim in claims]

        # Verify claims
        logger.info("Verifying claims...")
        verified_claims = self.verify_claims(extracted_claims, document)

        # Organize results
        supported_claims = [c for c in verified_claims if c["verification"]["verdict"] == "SUPPORTED"]
        refuted_claims = [c for c in verified_claims if c["verification"]["verdict"] == "REFUTED"]
        nei_claims = [c for c in verified_claims if c["verification"]["verdict"] == "NOT_ENOUGH_INFO"]

        logger.info(f"Results: {len(supported_claims)} supported, {len(refuted_claims)} refuted, {len(nei_claims)} undetermined")

        return {
            "all_claims": verified_claims,
            "supported_claims": supported_claims,
            "refuted_claims": refuted_claims,
            "undetermined_claims": nei_claims,
            "summary": {
                "total_claims": len(verified_claims),
                "supported_count": len(supported_claims),
                "refuted_count": len(refuted_claims),
                "undetermined_count": len(nei_claims)
            }
        }

    def format_results(self, results: Dict) -> str:
        """Format verification results as a readable text document."""
        lines = ["# Claim Verification Results\n"]

        # Summary statistics
        summary = results["summary"]
        lines.append(f"## Summary\n")
        lines.append(f"- Total claims analyzed: {summary['total_claims']}")
        lines.append(f"- Supported claims: {summary['supported_count']}")
        lines.append(f"- Refuted claims: {summary['refuted_count']}")
        lines.append(f"- Undetermined claims: {summary['undetermined_count']}\n")

        # Supported claims
        lines.append(f"## Supported Claims\n")
        if results["supported_claims"]:
            for i, claim in enumerate(results["supported_claims"]):
                lines.append(f"### Claim {i+1}: ✅ SUPPORTED")
                lines.append(f"**Claim text**: \"{claim['text']}\"")
                lines.append(f"**Confidence**: {claim['verification']['confidence']:.2f}")
                lines.append(f"**Supporting evidence**: \"{claim['verification']['evidence']}\"\n")
        else:
            lines.append("No supported claims found.\n")

        # Refuted claims
        lines.append(f"## Refuted Claims\n")
        if results["refuted_claims"]:
            for i, claim in enumerate(results["refuted_claims"]):
                lines.append(f"### Claim {i+1}: ❌ REFUTED")
                lines.append(f"**Claim text**: \"{claim['text']}\"")
                lines.append(f"**Confidence**: {claim['verification']['confidence']:.2f}")
                lines.append(f"**Contradicting evidence**: \"{claim['verification']['evidence']}\"\n")
        else:
            lines.append("No refuted claims found.\n")

        # Undetermined claims
        lines.append(f"## Undetermined Claims\n")
        if results["undetermined_claims"]:
            for i, claim in enumerate(results["undetermined_claims"]):
                lines.append(f"### Claim {i+1}: ❓ NOT ENOUGH INFO")
                lines.append(f"**Claim text**: \"{claim['text']}\"")
                if claim['verification']['evidence']:
                    lines.append(f"**Closest evidence found**: \"{claim['verification']['evidence']}\"\n")
                else:
                    lines.append("**No relevant evidence found**\n")
        else:
            lines.append("No undetermined claims found.\n")

        return "\n".join(lines)

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipe = ClaimExtractionVerificationPipeline()
    document = """
    The United Nations was founded on October 24, 1945 after World War II. Its headquarters are located in New York City.
    The current Secretary-General of the UN is António Guterres, who took office in 2017.
    The UN has 193 member states as of 2022, with the latest member being South Sudan which joined in 2011.
    The UN Security Council has 15 members, including 5 permanent members with veto power: China, France, Russia, the United Kingdom, and the United States.
    The UN General Assembly meets in regular sessions beginning on the third Tuesday in September.
    """
    app.run(host="0.0.0.0", port=5050, debug=True)
    # Example document
    

    # Example claims to verify (these would normally be extracted from the document)
    claims =  "The United Nations was founded in 1945.The UN headquarters are in Washington DC.António Guterres is the Secretary-General of the UN.The UN has 200 member states.The Security Council has 5 permanent members."
    

    # Process the document with provided claims
    results = pipeline.process_document(document, claims)

    # Format results
    formatted_results = pipeline.format_results(results)
    print(formatted_results)

    # Example of extracting claims directly from the document
    # results = pipeline.process_document(document)
    # formatted_results = pipeline.format_results(results)
    # print(formatted_results)