import React from "react";
import "./LandingPage.css";
import { useNavigate } from "react-router-dom";

function LandingPage() {

  const navigate = useNavigate();   // navigation hook

  return (
    <div className="landing-container">
      
      {/* Navbar */}
      <nav className="navbar">
        <div className="logo">VerifyAI</div>

        

        {/* Get Started Button */}
        <button 
          className="start-btn"
          onClick={() => navigate("/login")}
        >
          Get Started
        </button>

      </nav>


      {/* Hero Section */}
      <div className="hero">

        <div className="badge">⚡ AI-Powered KYC Compliance</div>

        <h1 className="hero-title">
          Identity Verification
          <br />
          <span>Reimagined with AI</span>
        </h1>

        <p className="hero-desc">
          Advanced fraud detection for BFSI KYC & AML compliance using
          Graph Neural Networks, NLP, and Computer Vision — powered by
          Azure OpenAI.
        </p>

        <div className="hero-buttons">

          <button 
            className="verify-btn"
            onClick={() => navigate("/login")}
          >
            Start Verification →
          </button>
          <button className="learn-btn" onClick={() => navigate("/learn-more")}>
             Learn More
            </button>

          

        </div>

      </div>

    </div>
  );
}

export default LandingPage;