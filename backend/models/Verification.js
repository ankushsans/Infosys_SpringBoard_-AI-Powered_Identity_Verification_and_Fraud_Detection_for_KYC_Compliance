const mongoose = require('mongoose');

const verificationSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.ObjectId,
    ref: 'User',
    required: true
  },
  name: {
    type: String,
    required: true
  },
  initials: String,
  docType: {
    type: String,
    required: true
  },
  docTypeSub: String,
  date: {
    type: Date,
    default: Date.now
  },
  confidence: {
    type: Number,
    required: true
  },
  status: {
    type: String,
    enum: ['Approved', 'Pending', 'Rejected'],
    default: 'Pending'
  },
  details: {
    faceMatch: String,
    forgeryDetection: String,
    documentStatus: String,
    extractedName: String,
    extractedAddress: String
  },
  identityFile: String,
  supportingFiles: [String]
});

module.exports = mongoose.model('Verification', verificationSchema);
