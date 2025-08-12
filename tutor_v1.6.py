import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
load_dontenv()


class ConversationManager:
    def __init__(self, max_history: int = 5):
        """Manage conversation history for context-aware responses"""
        self.history: List[Tuple[str, str]] = []  # [(question, answer), ...]
        self.max_history = max_history

    def add_exchange(self, question: str, answer: str):
        """Add a Q&A exchange to history"""
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history.pop(0)  # Keep only recent exchanges

    def get_context(self) -> str:
        """Format conversation history for OpenAI prompt"""
        if not self.history:
            return ""

        context = "Previous conversation:\n"
        for i, (q, a) in enumerate(self.history, 1):
            context += f"Q{i}: {q}\n"
            context += f"A{i}: {a}\n\n"

        return context

    def clear_history(self):
        """Clear conversation history"""
        self.history = []


class PhysicsRAGSystem:
    def __init__(self, db_path: str = os.getenv("OPENAI_API_KEY"),
                 openai_api_key: str = os.getenv("OPENAI_API_KEY")):
        """Initialize the Physics RAG system"""
        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="physics_knowledge",
            metadata={"description": "Physics Knowledge Base"}
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

        # Initialize conversation manager
        self.conversation = ConversationManager()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except:
                        continue
                return text
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # Try to break at sentence boundary
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

        return chunks

    def add_content(self, text: str, metadata: Dict, doc_id: str) -> bool:
        """Add content to the database"""
        try:
            embedding = self.embedding_model.encode([text])
            self.collection.add(
                embeddings=embedding.tolist(),
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Error adding content {doc_id}: {str(e)}")
            return False

    def search(self, query: str, n_results: int = 3, filters: Optional[Dict] = None) -> Dict:
        """Search for relevant content"""
        try:
            query_embedding = self.embedding_model.encode([query])
            search_kwargs = {
                "query_embeddings": query_embedding.tolist(),
                "n_results": n_results
            }
            if filters:
                search_kwargs["where"] = filters

            results = self.collection.query(**search_kwargs)
            return {
                "documents": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}

    def doubt_clarification(self, query: str) -> dict:
        """Handle doubt clarification - reactive, specific answers"""
        search_results = self.search(query, n_results=3)

        if not search_results['documents']:
            return {
                "answer": "I couldn't find relevant information for your question. Please try rephrasing or ask about a different physics topic.",
                "follow_up_questions": [],
                "sources": []
            }

        context = "\n\n".join(search_results['documents'][:3])

        # Add conversation context if available
        conversation_context = self.conversation.get_context()
        if conversation_context:
            prompt = f"""You are a physics tutor answering a student doubt. Be direct and concise.

{conversation_context}

Context from textbooks:
{context}

Current Student Question: {query}

Provide a clear, focused answer. If the question references previous discussion, acknowledge it appropriately."""
        else:
            prompt = f"""You are a physics tutor answering a specific student doubt. Be direct and concise.

Context from textbooks:
{context}

Student Question: {query}

Provide a clear, focused answer to this specific question. Assume the student has some basic knowledge."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful physics tutor. Answer specific questions directly and clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )

            answer = response.choices[0].message.content
            follow_ups = self.generate_follow_ups(query, answer)
            sources = self.extract_sources(search_results['metadatas'])

            # Store this exchange in conversation history
            self.conversation.add_exchange(query, answer)

            return {
                "answer": answer,
                "follow_up_questions": follow_ups,
                "sources": sources
            }
        except Exception as e:
            return {"error": f"Error generating response: {str(e)}"}

    def learning_assistant(self, topic: str) -> dict:
        """Handle learning assistance - proactive, structured explanations"""
        # Search for comprehensive content on the topic
        search_results = self.search(topic, n_results=5)

        if not search_results['documents']:
            return {
                "explanation": f"I couldn't find information about '{topic}'. Please try a different physics topic.",
                "sources": []
            }

        context = "\n\n".join(search_results['documents'][:5])

        # Add conversation context if available
        conversation_context = self.conversation.get_context()
        if conversation_context:
            prompt = f"""You are a physics teacher explaining a topic from scratch. Provide a comprehensive, structured explanation.

{conversation_context}

Context from textbooks:
{context}

Current Topic to explain: {topic}

Structure your explanation as follows:
1. INTRODUCTION: Brief overview of what this topic is about
2. KEY CONCEPTS: Break down the fundamental concepts step by step
3. EXAMPLES: Provide clear, practical examples
4. APPLICATIONS: Explain real-world applications or uses
5. SUMMARY: Summarize the key takeaways

If this topic relates to previous discussion, acknowledge the connection. Assume the student is learning this for the first time. Be thorough but clear."""
        else:
            prompt = f"""You are a physics teacher explaining a topic from scratch. Provide a comprehensive, structured explanation.

Context from textbooks:
{context}

Topic to explain: {topic}

Structure your explanation as follows:
1. INTRODUCTION: Brief overview of what this topic is about
2. KEY CONCEPTS: Break down the fundamental concepts step by step
3. EXAMPLES: Provide clear, practical examples
4. APPLICATIONS: Explain real-world applications or uses
5. SUMMARY: Summarize the key takeaways

Assume the student is learning this for the first time. Be thorough but clear."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a patient physics teacher providing structured, comprehensive explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )

            explanation = response.choices[0].message.content
            sources = self.extract_sources(search_results['metadatas'])

            # Store this exchange in conversation history
            self.conversation.add_exchange(f"Explain: {topic}", explanation)

            return {
                "explanation": explanation,
                "sources": sources
            }
        except Exception as e:
            return {"error": f"Error generating explanation: {str(e)}"}

    def quiz_generation(self, topic: str, num_questions: int = 5, quiz_mode: str = "mixed") -> dict:
        """Generate quiz questions on a topic"""
        # Different search strategies based on mode
        if quiz_mode.lower() == "jee":
            # Prioritize JEE question bank, then reference books
            search_results_primary = self.search(f"{topic} problems questions", n_results=6,
                                                 filters={"source_type": "jee_questions"})
            search_results_secondary = self.search(f"{topic} numerical calculations", n_results=2,
                                                   filters={"source_type": "jee"})

            # Combine results, prioritizing question bank
            all_docs = search_results_primary['documents'] + search_results_secondary['documents']
            all_metas = search_results_primary['metadatas'] + search_results_secondary['metadatas']

        elif quiz_mode.lower() == "ncert":
            # Focus on NCERT content
            search_results = self.search(f"{topic} concepts questions examples", n_results=6,
                                         filters={"source_type": "ncert"})
            all_docs = search_results['documents']
            all_metas = search_results['metadatas']

        else:  # mixed
            search_results = self.search(f"{topic} problems questions", n_results=8)
            all_docs = search_results['documents']
            all_metas = search_results['metadatas']

        if not all_docs:
            return {
                "error": f"Couldn't find enough content to generate {quiz_mode} quiz on '{topic}'"
            }

        context = "\n\n".join(all_docs[:6])

        # Mode-specific prompting
        if quiz_mode.lower() == "jee":
            mode_instruction = """Focus on:
- 70-80% numerical/calculation-based problems requiring formulas and multi-step solutions
- 20-30% challenging conceptual questions that require deep understanding and analysis
- Questions similar to JEE Main exam style
- Include problems with specific values, units, and calculations"""

        elif quiz_mode.lower() == "ncert":
            mode_instruction = """Focus on:
- Conceptual understanding questions
- Basic applications of principles
- Straightforward problems from NCERT textbooks
- Definition and explanation based questions"""

        else:  # mixed
            mode_instruction = "Mix of conceptual and numerical questions from all sources."

        prompt = f"""You are creating a physics quiz. Generate exactly {num_questions} multiple choice questions.

Context from textbooks:
{context}

Topic: {topic}
Mode: {quiz_mode.upper()}

{mode_instruction}

For each question, provide:
1. Question text
2. Four options (A, B, C, D)  
3. Correct answer (A, B, C, or D)
4. Brief explanation of why the answer is correct

Format as:
Q1: [Question]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
Answer: [Letter]
Explanation: [Brief explanation]

IMPORTANT: If a question requires diagrams, graphs, or tables that may be missing from the text, 
add a note: "Refer to original source for diagrams/tables if needed."

Generate exactly {num_questions} questions."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a physics teacher creating quiz questions. Follow the exact format requested."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.4
            )

            quiz_content = response.choices[0].message.content
            questions = self.parse_quiz(quiz_content)
            sources = self.extract_sources(all_metas)

            return {
                "questions": questions,
                "total_questions": len(questions),
                "topic": topic,
                "quiz_mode": quiz_mode,
                "sources": sources
            }
        except Exception as e:
            return {"error": f"Error generating quiz: {str(e)}"}

    def parse_quiz(self, quiz_content: str) -> List[Dict]:
        """Parse quiz content into structured format"""
        questions = []
        current_question = {}

        lines = quiz_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_question:
                    questions.append(current_question)
                current_question = {"question": line.split(':', 1)[1].strip(), "options": {}}
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                option_letter = line[0]
                option_text = line[3:].strip()
                current_question["options"][option_letter] = option_text
            elif line.startswith('Answer:'):
                current_question["correct_answer"] = line.split(':', 1)[1].strip()
            elif line.startswith('Explanation:'):
                current_question["explanation"] = line.split(':', 1)[1].strip()

        if current_question:
            questions.append(current_question)

        return questions

    def generate_follow_ups(self, original_query: str, answer: str) -> List[str]:
        """Generate follow-up questions for doubt clarification"""
        try:
            prompt = f"""Based on this physics Q&A, suggest 3 follow-up questions:

Question: {original_query}
Answer: {answer}

Generate 3 follow-up questions:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.5
            )

            follow_ups_text = response.choices[0].message.content
            follow_ups = [q.strip() for q in follow_ups_text.split('\n') if q.strip() and '?' in q]
            return follow_ups[:3]
        except:
            return [
                "Can you give me an example?",
                "How is this used in real life?",
                "What are some practice problems?"
            ]

    def extract_sources(self, metadatas: List[Dict]) -> List[str]:
        """Extract source information from metadata"""
        sources = []
        for meta in metadatas:
            if meta.get('source_type') == 'ncert':
                class_num = meta.get('class', 0)
                part_num = meta.get('part', 0)
                if class_num > 0 and part_num > 0:
                    source_info = f"NCERT Class {class_num} Part {part_num}"
                else:
                    source_info = "NCERT Textbook"
            elif meta.get('source_type') == 'jee':
                book_name = meta.get('book_name', 'JEE Reference')
                source_info = f"{book_name}"
            elif meta.get('source_type') == 'jee_questions':
                book_name = meta.get('book_name', 'JEE Question Bank')
                source_info = f"{book_name}"
            else:
                source_info = "Physics Textbook"

            if source_info not in sources:
                sources.append(source_info)

        return sources

    def process_book(self, pdf_path: str, book_info: Dict) -> bool:
        """Process a single book"""
        print(f"Processing: {book_info.get('name', 'Unknown Book')}")

        # Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            print("No text extracted")
            return False

        # Clean and chunk text
        full_text = ' '.join(full_text.split())
        chunks = self.chunk_text(full_text, chunk_size=800, overlap=200)

        # Add chunks to database
        added_count = 0
        for i, chunk in enumerate(chunks):
            doc_id = f"{book_info['source_type']}_{book_info.get('identifier', 'unknown')}_{i + 1:03d}"

            metadata = {
                "source_type": book_info['source_type'],
                "book_name": book_info.get('name', 'Unknown'),
                "class": book_info.get('class', 0),
                "part": book_info.get('part', 0),
                "author": book_info.get('author', 'Unknown'),
                "chunk_number": i + 1,
                "total_chunks": len(chunks)
            }

            if self.add_content(chunk, metadata, doc_id):
                added_count += 1

        print(f"Added {added_count}/{len(chunks)} chunks")
        return True

    def get_stats(self):
        """Get database statistics"""
        try:
            return {"total_documents": self.collection.count()}
        except:
            return {"total_documents": 0}

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation.clear_history()
        print("Conversation history cleared!")


def clear_database():
    """Clear the database"""
    try:
        client = chromadb.PersistentClient(path="./physics_knowledge_db")
        try:
            client.delete_collection("physics_knowledge")
            print("Database cleared")
        except:
            print("No existing database found")
        return True
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False


def setup_books(rag_system, folder_path: str):
    """Setup all books in the system"""
    if not os.path.exists(folder_path):
        print("Folder not found")
        return False

    # NCERT Books
    ncert_books = [
        {"file": "Class11_Part1.pdf", "name": "NCERT Class 11 Part 1", "source_type": "ncert", "class": 11, "part": 1,
         "identifier": "ncert_11_1"},
        {"file": "Class11_Part2.pdf", "name": "NCERT Class 11 Part 2", "source_type": "ncert", "class": 11, "part": 2,
         "identifier": "ncert_11_2"},
        {"file": "Class12_Part1.pdf", "name": "NCERT Class 12 Part 1", "source_type": "ncert", "class": 12, "part": 1,
         "identifier": "ncert_12_1"},
        {"file": "Class12_Part2.pdf", "name": "NCERT Class 12 Part 2", "source_type": "ncert", "class": 12, "part": 2,
         "identifier": "ncert_12_2"}
    ]

    # JEE Reference Books
    jee_books = [
        {"file": "DC_Pandey_Mechanics_Volume_2.pdf", "name": "DC Pandey Mechanics Vol 2", "source_type": "jee",
         "author": "DC Pandey", "identifier": "dc_mech2"},
        {"file": "DC_Pandey_Volume 3_ Electricity_and_Magnetism.pdf", "name": "DC Pandey Electricity & Magnetism",
         "source_type": "jee", "author": "DC Pandey", "identifier": "dc_elec"},
        {"file": "DC_Pandey_Optics_And_Modern_Physics.pdf", "name": "DC Pandey Optics & Modern Physics",
         "source_type": "jee", "author": "DC Pandey", "identifier": "dc_optics"},
        {"file": "DC_Pandey_Waves_And_Thermodynamics.pdf", "name": "DC Pandey Waves & Thermodynamics",
         "source_type": "jee", "author": "DC Pandey", "identifier": "dc_waves"},
        {"file": "HC_Verma_Part1.pdf", "name": "HC Verma Concepts of Physics Vol 1", "source_type": "jee",
         "author": "HC Verma", "identifier": "hc_vol1"},
        {"file": "HC_Verma_Part2.pdf", "name": "HC Verma Concepts of Physics Vol 2", "source_type": "jee",
         "author": "HC Verma", "identifier": "hc_vol2"}
    ]

    all_books = ncert_books + jee_books
    processed = 0

    for book in all_books:
        pdf_path = os.path.join(folder_path, book["file"])
        if os.path.exists(pdf_path):
            if rag_system.process_book(pdf_path, book):
                processed += 1
        else:
            print(f"File not found: {book['file']}")

    print(f"Setup complete. Processed {processed}/{len(all_books)} books.")
    return processed > 0


def demo():
    """Main demo function"""
    print("🚀 Physics AI Tutor Demo")
    print("=" * 40)

    # Check if we need fresh setup
    setup_needed = input("Setup books from folder? (y/n): ").lower() == 'y'

    if setup_needed:
        clear_db = input("Clear existing database? (y/n): ").lower() == 'y'
        if clear_db:
            clear_database()

    # Initialize system
    rag = PhysicsRAGSystem()

    if setup_needed:
        folder_path = input("Enter folder path with PDF books: ").strip().strip('"')
        setup_books(rag, folder_path)

    # Show current stats
    stats = rag.get_stats()
    print(f"\nDatabase has {stats['total_documents']} documents")

    if stats['total_documents'] == 0:
        print("No content in database. Please setup books first.")
        return

    # Demo loop
    print("\n🎓 Welcome to Physics AI Tutor!")
    print("Available modes:")
    print("1. 🤔 Doubt Clarification - Ask specific questions")
    print("2. 📚 Learning Assistant - Learn topics step-by-step")
    print("3. 📝 Quiz Generation - Generate practice questions")

    while True:
        print("\n" + "-" * 50)
        print("Choose mode:")
        print("1. Doubt Clarification")
        print("2. Learning Assistant")
        print("3. Quiz Generation")
        print("4. Exit")

        choice = input("Enter choice (1-4): ").strip()

        if choice == '4':
            break
        elif choice == '1':
            print("\n🤔 Doubt Clarification Mode")
            print("Ask your physics questions naturally. Type 'exit' to return to main menu.")

            while True:
                question = input("\nQuestion (type exit to return to mode selection): ").strip()
                if question.lower() == 'exit':
                    rag.clear_conversation()
                    break

                if question:
                    print("\nThinking...")
                    result = rag.doubt_clarification(question)

                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\n📖 Answer:")
                        print(result['answer'])

                        if result['follow_up_questions']:
                            print(f"\n🔄 Suggested follow-ups:")
                            for i, follow_up in enumerate(result['follow_up_questions'], 1):
                                print(f"  {i}. {follow_up}")

                        if result['sources']:
                            print(f"\n📚 Sources: {', '.join(result['sources'])}")

        elif choice == '2':
            print("\n📚 Learning Assistant Mode")
            print("Ask me to explain physics topics. Type 'exit' to return to main menu.")

            while True:
                topic = input("\nTopic to learn (or question, type exit to return to switch modes): ").strip()
                if topic.lower() == 'exit':
                    rag.clear_conversation()
                    break

                if topic:
                    print("\nPreparing explanation...")
                    result = rag.learning_assistant(topic)

                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\n📖 Comprehensive Explanation:")
                        print(result['explanation'])

                        if result['sources']:
                            print(f"\n📚 Sources: {', '.join(result['sources'])}")

        elif choice == '3':
            print("\n📝 Quiz Generation Mode")
            print("Generate practice questions. Type 'exit' to return to main menu.")

            while True:
                topic = input("\nTopic for quiz (or 'exit'): ").strip()
                if topic.lower() == 'exit':
                    break

                if topic:
                    print("\nQuiz modes:")
                    print("1. NCERT - Conceptual questions from textbooks")
                    print("2. JEE - Numerical & challenging problems")
                    print("3. Mixed - All sources")

                    mode_choice = input("Choose mode (1-3): ").strip()
                    quiz_mode = {"1": "ncert", "2": "jee", "3": "mixed"}.get(mode_choice, "mixed")

                    num_q = input("Number of questions (default 5): ").strip()
                    num_questions = int(num_q) if num_q.isdigit() else 5

                    print(f"\nGenerating {quiz_mode.upper()} quiz...")
                    result = rag.quiz_generation(topic, num_questions, quiz_mode)

                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\n📝 Quiz: {result['topic']} ({result['quiz_mode'].upper()} mode)")
                        print("=" * 50)

                        for i, q in enumerate(result['questions'], 1):
                            print(f"\nQ{i}: {q['question']}")
                            for letter, option in q['options'].items():
                                print(f"{letter}) {option}")
                            print(f"Answer: {q['correct_answer']}")
                            print(f"Explanation: {q['explanation']}")
                            print("-" * 30)

                        if result['sources']:
                            print(f"\n📚 Sources: {', '.join(result['sources'])}")

        else:
            print("Invalid choice. Please enter 1-4.")

    print("\nThanks for using Physics AI Tutor!")


if __name__ == "__main__":
    demo()