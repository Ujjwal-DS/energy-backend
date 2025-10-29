#  Energy Backend (FastAPI)

This is the backend service for the GH2 / Energy optimization project.  
It provides REST API endpoints to process renewable energy data, run optimization logic, and generate outputs dynamically.

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Ujjwal-DS/energy-backend.git
cd energy-backend

2. Create a Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # for Windows PowerShell

3. Install Dependencies
pip install -r requirements.txt

4. Setup Input Data

This backend requires static CSV files in the input/ folder (mapping, assumptions, etc.).
These are not included in the repo.
Ask Ujjwal for input_data.zip, then unzip it inside the project root:

energy-backend/
  ├─ app/
  ├─ input/
  ├─ output/
  ├─ requirements.txt
  └─ README.md

5. Run the FastAPI Server
uvicorn app.main:app --reload --port 8000


Open http://127.0.0.1:8000/docs
 to view Swagger UI.

⚙️ Project Structure
energy-backend/
├─ app/
│  ├─ api.py
│  ├─ config.py
│  ├─ schemas.py
│  ├─ ...
├─ helpers.py
├─ logic.py
├─ main.py
├─ preprocessing.py
├─ postprocessing.py
├─ utils.py
├─ requirements.txt
├─ .gitignore
└─ README.md

👥 Frontend Integration Notes

FastAPI endpoints are CORS-enabled.

Run frontend (e.g., React @ port 3000) and backend (port 8000) locally.

🧑‍💻 Contributor Workflow
git pull origin main
# make changes
git add .
git commit -m "Describe changes"
git push origin main


Do not commit input/ or output/ — they’re ignored intentionally.

📞 Contact

Ujjwal Gupta – Data Scientist