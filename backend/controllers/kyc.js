const path = require('path');
const fs = require('fs');
const Verification = require('../models/Verification');

exports.verifyDocuments = async (req, res, next) => {
  try {
    if (!req.files || (!req.files['identity'] && !req.files['supporting'])) {
      return res.status(400).json({
        success: false,
        error: 'Please upload at least one document'
      });
    }

    const identityFile = req.files['identity'] ? req.files['identity'][0] : null;
    const supportingFiles = req.files['supporting'] || [];

    // Simulate ML Processing delay
    // Mocking the result
    const mockResult = {
      status: 'Approved',
      confidenceScore: 0.92,
      details: {
        faceMatch: 'Successful',
        forgeryDetection: 'Low Risk',
        documentStatus: 'Verified',
        extractedName: 'John Doe',
        extractedAddress: '123 Main St, Springfield'
      }
    };

    // Save to database
    const verification = await Verification.create({
      user: req.user.id,
      name: req.user.name,
      initials: req.user.name.split(' ').map(n => n[0]).join(''),
      docType: identityFile ? 'Aadhaar Card' : 'Supporting document',
      docTypeSub: identityFile ? 'Aadhaar' : 'Misc',
      confidence: mockResult.confidenceScore * 100,
      status: mockResult.status,
      details: mockResult.details,
      identityFile: identityFile ? identityFile.path : null,
      supportingFiles: supportingFiles.map(f => f.path)
    });

    res.status(200).json({
      success: true,
      data: verification
    });

  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message
    });
  }
};

exports.getHistory = async (req, res, next) => {
  try {
    const verifications = await Verification.find({ user: req.user.id }).sort('-date');

    res.status(200).json({
      success: true,
      count: verifications.length,
      data: verifications
    });
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message
    });
  }
};
