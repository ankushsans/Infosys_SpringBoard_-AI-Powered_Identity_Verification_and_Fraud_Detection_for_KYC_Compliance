# AI-Powered KYC & AML Compliance System

An end-to-end AI system for identity document verification, address validation, and fraud detection for BFSI (Banking, Financial Services, and Insurance) workflows. This project leverages Computer Vision, NLP, and a robust MERN stack backend to automate KYC compliance.

## Features

- **User Authentication**: Secure registration and login using JWT and Bcrypt.
- **AI Document Classification**: Automatically detects and classifies Aadhaar, PAN, and Passports.
- **Document Verification**: Upload and process identity documents with real-time feedback.
- **KYC History**: Track all past verification attempts with detailed status and results.
- **OCR Integration**: (In Progress) Text extraction for automated data entry.
- **Modern Dashboard**: Responsive UI with glassmorphism and real-time analysis insights.

## Project Structure

- `frontend/`: React + Vite application (User Interface).
- `backend/`: Node.js + Express + MongoDB (API Service).
- `notebooks/`: AI/ML research and model training (TensorFlow/Keras).

## Tech Stack

- **Frontend**: React.js, React Router, Vanilla CSS, Vite.
- **Backend**: Node.js, Express.js, MongoDB (Mongoose), JWT, Multer.
- **ML/AI**: TensorFlow, Keras, OpenCV, Gradio, OpenBharatOCR.

---

## Getting Started

### 1. Prerequisites
- Node.js (v18+)
- MongoDB (Running locally or Atlas cloud instance)
- Python 3.9+ (For ML notebooks)

### 2. Backend Setup
```bash
cd backend
npm install
```
Create a `.env` file in the `backend/` directory:
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/kyc_db
JWT_SECRET=your_super_secret_key
NODE_ENV=development
```
Start the backend server:
```bash
npm run dev
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🛰️ API Endpoints

### Authentication
- `POST /api/auth/register` - Create a new account.
- `POST /api/auth/login` - Authenticate user and get token.

### KYC Operations (Requires Auth)
- `POST /api/kyc/verify` - Upload identity and supporting documents.
- `GET /api/kyc/history` - Retrieve a user's verification history.

---

## ML Integration
The classification models are located in `notebooks/`. 
To run the ML service interface:
1. Navigate to `notebooks/`.
2. Open `KYC Identification for Aadhar, Pan and Passport.ipynb`.
3. Run the Gradio interface cells to start the prediction API.

## Status & Roadmap
- [x] Basic Auth and User Management
- [x] Document Upload & Storage Logic
- [x] Batch Classification of ID Documents
- [x] History Tracking Dashboard
- [ ] Live ML Service Endpoint Integration
- [ ] Automated Address Matching Logic
- [ ] Forgery Detection Module
