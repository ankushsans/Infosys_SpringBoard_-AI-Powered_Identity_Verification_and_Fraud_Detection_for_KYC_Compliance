import { Routes, Route } from "react-router-dom";

import LandingPage from "./components/Landingpage";
import Login from "./components/Login";
import Register from "./components/Register";
import Upload from "./components/Upload";
import ChatHistory from "./components/ChatHistory";
import Account from "./components/Account";
import LearnMore from "./components/Learnmore";   // ADD THIS

import "./styles/dashboard.css";

function App() {
  return (
    <Routes>

      {/* Landing Page */}
      <Route path="/" element={<LandingPage />} />

      {/* Learn More Page */}
      <Route path="/learn-more" element={<LearnMore />} />

      {/* Authentication */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* Dashboard Pages */}
      <Route path="/upload" element={<Upload />} />
      <Route path="/chat-history" element={<ChatHistory />} />
      <Route path="/account" element={<Account />} />

    </Routes>
  );
}

export default App;