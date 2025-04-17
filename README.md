RAG-Based Smart Advisor System in Persian
This project is an AI-based system leveraging RAG (Retrieval-Augmented Generation) for academic advising and study planning in Persian. The system can utilize your PDF files as a knowledge source to answer questions and generate personalized study schedules.

Features
Full support for the Persian language
Processing and extracting information from PDF files
Smart advising using RAG technology
Generating weekly study plans for students
Web-based user interface powered by Streamlit
Requirements
Python 3.8 or higher
Required packages listed in the requirements.txt file
Installation and Setup
Clone the repository or download the files:
bash



git clone <repository-url>
cd <project-folder-name>
Create a virtual environment (optional but recommended):
bash



python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
Install dependencies:
bash



pip install -r requirements.txt
Run the application:
bash



streamlit run app.py
Your browser will open automatically. If it doesn’t, copy the URL displayed in the terminal and open it in your browser.
Project Structure
app.py: Main Streamlit application file
pdf_processor.py: PDF processing and text extraction
rag_manager.py: RAG system and language model management
study_planner.py: Weekly study plan generation
data/: Folder for storing PDF files
vectordb/: Folder for storing the vector database
How to Use
1. Uploading PDF Files
In the sidebar, click on "Upload PDF Files."
Select your PDF files.
Click the "Process Files" button.
Wait for the files to be processed.
2. Interacting with the Smart Advisor
In the "Advising and Chat" section, type your question in the text box at the bottom of the page.
The system will respond using the information from the processed PDF files.
3. Creating a Weekly Study Plan
In the sidebar, click on "Weekly Study Plan."
Fill out the form with the student’s information.
Enter the subjects and their priorities.
Click the "Generate Weekly Plan" button.
View the generated plan and download it if needed.
Changing the Language Model
To change the language model, open the rag_manager.py file and edit the get_llm() function. You can replace it with other models from Hugging Face or different sources.

Important Notes
For optimal performance, use high-quality PDF files.
If you have a GPU, the system will automatically utilize it.
Larger models may require more powerful hardware.
Contribution
Your contributions to improve this project are greatly appreciated. Please report issues or submit pull requests.
