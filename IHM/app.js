// Configuration
const API_URL = 'http://localhost:5000';

// State
let currentFile = null;
let currentResult = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadContent = document.getElementById('uploadContent');
const previewArea = document.getElementById('previewArea');
const previewImg = document.getElementById('previewImg');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultsContent = document.getElementById('resultsContent');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    checkAPIStatus();
    setInterval(checkAPIStatus, 30000);
});

// API Status
async function checkAPIStatus() {
    const statusIndicator = document.getElementById('apiStatus');
    const statusText = document.getElementById('apiStatusText');
    
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusIndicator.classList.add('online');
            statusIndicator.classList.remove('offline');
            statusText.textContent = 'Online';
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        statusIndicator.classList.add('offline');
        statusIndicator.classList.remove('online');
        statusText.textContent = 'Offline';
    }
}

// Upload
function initUpload() {
    // Browse button
    browseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Browse button clicked');
        fileInput.click();
    });
    
    // Upload area click
    uploadArea.addEventListener('click', (e) => {
        if (e.target === uploadArea || e.target.closest('.upload-content')) {
            console.log('Upload area clicked');
            fileInput.click();
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        console.log('File input changed');
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name, file.type, file.size);
            handleFile(file);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Dragover');
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Dragleave');
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Drop event');
        uploadArea.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        console.log('Dropped file:', file);
        
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        } else {
            alert('Please drop an image file');
        }
    });
    
    // Remove button
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        console.log('Remove button clicked');
        resetUpload();
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', () => {
        console.log('Analyze button clicked');
        analyzeImage();
    });
    
    // New analysis
    document.getElementById('newAnalysisBtn').addEventListener('click', () => {
        console.log('New analysis button clicked');
        resetUpload();
        showResults(false);
    });
    
    // Download
    document.getElementById('downloadBtn').addEventListener('click', () => {
        console.log('Download button clicked');
        downloadReport();
    });
}

function handleFile(file) {
    console.log('handleFile called with:', file.name);
    
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    currentFile = file;
    console.log('Current file set:', currentFile.name);
    
    const reader = new FileReader();
    reader.onload = (e) => {
        console.log('File loaded, displaying preview');
        previewImg.src = e.target.result;
        uploadContent.style.display = 'none';
        previewArea.style.display = 'block';
        analyzeBtn.disabled = false;
        console.log('Preview displayed, analyze button enabled');
    };
    reader.onerror = (e) => {
        console.error('Error reading file:', e);
        alert('Error reading file');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    uploadContent.style.display = 'block';
    previewArea.style.display = 'none';
    previewImg.src = '';
    analyzeBtn.disabled = true;
}

// Analysis
async function analyzeImage() {
    if (!currentFile) return;
    
    showResults(false);
    emptyState.style.display = 'none';
    loadingState.style.display = 'block';
    resultsContent.style.display = 'none';
    analyzeBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('image', currentFile);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        
        const data = await response.json();
        if (data.success) {
            currentResult = data.result;
            displayResults(data.result);
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Analysis failed. Please check if the API is running.');
        showResults(false);
        emptyState.style.display = 'block';
        analyzeBtn.disabled = false;
    } finally {
        loadingState.style.display = 'none';
    }
}

function displayResults(result) {
    document.getElementById('diagnosisName').textContent = result.prediction;
    document.getElementById('confidenceText').textContent = `${result.confidence.toFixed(1)}%`;
    
    const confidenceBar = document.getElementById('confidenceBar');
    setTimeout(() => {
        confidenceBar.style.width = `${result.confidence}%`;
    }, 100);
    
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = result.risk_level;
    riskBadge.className = `risk-badge ${result.risk_level}`;
    
    document.getElementById('clinicalNote').textContent = result.clinical_note;
    document.getElementById('recommendedAction').textContent = result.recommended_action;
    
    displayProbabilities(result.probabilities);
    showResults(true);
}

function displayProbabilities(probabilities) {
    const container = document.getElementById('probabilityBars');
    container.innerHTML = '';
    
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sorted.forEach(([className, probability]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        item.innerHTML = `
            <div class="probability-label">
                <strong>${className}</strong>
                <span class="probability-value">${probability.toFixed(1)}%</span>
            </div>
            <div class="probability-bar-bg">
                <div class="probability-bar-fill" style="width: ${probability}%">
                    ${probability > 5 ? probability.toFixed(1) + '%' : ''}
                </div>
            </div>
        `;
        
        container.appendChild(item);
    });
}

function showResults(show) {
    if (show) {
        emptyState.style.display = 'none';
        loadingState.style.display = 'none';
        resultsContent.style.display = 'block';
    } else {
        emptyState.style.display = 'block';
        loadingState.style.display = 'none';
        resultsContent.style.display = 'none';
    }
}

// Download Report
function downloadReport() {
    if (!currentResult) return;
    
    const report = generateTextReport(currentResult);
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `OCT_Analysis_${currentResult.metadata.timestamp}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function generateTextReport(result) {
    const lines = [];
    lines.push('='.repeat(60));
    lines.push('OCT IMAGE ANALYSIS REPORT');
    lines.push('='.repeat(60));
    lines.push('');
    lines.push(`Date: ${new Date().toLocaleString()}`);
    lines.push(`Image: ${result.metadata.filename}`);
    lines.push('');
    lines.push('PRIMARY DIAGNOSIS: ' + result.prediction);
    lines.push(`Confidence: ${result.confidence.toFixed(2)}%`);
    lines.push('');
    lines.push('-'.repeat(60));
    lines.push('RISK ASSESSMENT');
    lines.push('-'.repeat(60));
    lines.push(`Risk Level: ${result.risk_level}`);
    lines.push(`Recommended Action: ${result.recommended_action}`);
    lines.push('');
    lines.push('-'.repeat(60));
    lines.push('CLINICAL NOTES');
    lines.push('-'.repeat(60));
    lines.push(result.clinical_note);
    lines.push('');
    lines.push('-'.repeat(60));
    lines.push('PROBABILITY DISTRIBUTION');
    lines.push('-'.repeat(60));
    
    Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([cls, prob]) => {
            const bar = '█'.repeat(Math.floor(prob / 5));
            lines.push(`${cls.padEnd(10)}: ${prob.toFixed(2).padStart(6)}% ${bar}`);
        });
    
    lines.push('');
    lines.push('='.repeat(60));
    lines.push('DISCLAIMER: AI-assisted analysis for medical professional use.');
    lines.push('Final diagnosis must be confirmed by a qualified physician.');
    lines.push('='.repeat(60));
    
    return lines.join('\n');
}