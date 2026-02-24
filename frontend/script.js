const promptInput = document.getElementById('prompt-input');
const sendButton = document.getElementById('send-button');
const chatContainer = document.getElementById('chat-container');
const statusText = document.getElementById('status-text');

// Auto-resize textarea
promptInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value.trim().length > 0) {
        sendButton.style.opacity = '1';
    } else {
        sendButton.style.opacity = '0.5';
    }
});

promptInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener('click', sendMessage);

function addMessage(text, isUser = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-message' : 'system-message'}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Simple markdown formatting for bold and newlines
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formattedText = formattedText.replace(/\n/g, '<br>');
    bubble.innerHTML = formattedText;

    msgDiv.appendChild(bubble);
    chatContainer.appendChild(msgDiv);

    // Smooth scroll to bottom
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 50);

    return msgDiv; // Return reference to append badges later
}

function addTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-message typing-msg';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble typing-indicator';
    bubble.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;

    msgDiv.appendChild(bubble);
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return msgDiv;
}

function appendConfidenceBadge(messageElement, confidenceData) {
    const { score, level, clusters = 1 } = confidenceData;
    const percentage = (score * 100).toFixed(1);

    let badgeClass, iconLabel;
    if (level === 'high') {
        badgeClass = 'conf-high';
        iconLabel = 'High Confidence';
    } else if (level === 'medium') {
        badgeClass = 'conf-medium';
        iconLabel = 'Medium Confidence';
    } else {
        badgeClass = 'conf-low';
        iconLabel = 'Hallucination Risk';
    }

    const badge = document.createElement('div');
    badge.className = `confidence-badge ${badgeClass}`;
    badge.innerHTML = `
        <div class="dot"></div>
        <span>${iconLabel} (${percentage}%)</span>
        <span style="opacity: 0.6; font-weight: normal; text-transform: none; margin-left: 4px;">• ${clusters} Semantic Clusters</span>
    `;

    messageElement.appendChild(badge);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}


async function sendMessage() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Reset UI state
    promptInput.value = '';
    promptInput.style.height = 'auto';
    sendButton.disabled = true;

    addMessage(prompt, true);
    const indicator = addTypingIndicator();

    // Fun status updates since generation takes time
    let dots = 0;
    statusText.textContent = "Generating multiple diverse responses...";
    const interval = setInterval(() => {
        dots = (dots + 1) % 4;
        statusText.textContent = "Generating multiple diverse responses" + ".".repeat(dots);
    }, 500);

    setTimeout(() => { clearInterval(interval); statusText.textContent = "Running Semantic Verifier Model..."; }, 7000);
    setTimeout(() => { statusText.textContent = "Calculating Fermi-Dirac Logit Flows..."; }, 14000);

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, num_samples: 5 })
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();

        // Remove typing indicator
        indicator.remove();
        clearInterval(interval);
        statusText.textContent = "Ready";

        // Render Answer
        const msgEl = addMessage(data.answer, false);

        // Delay badge appearance slightly for dramatic effect
        setTimeout(() => {
            appendConfidenceBadge(msgEl, {
                score: data.confidence_score,
                level: data.confidence_level,
                clusters: data.clusters_found
            });
        }, 600);

    } catch (error) {
        indicator.remove();
        clearInterval(interval);
        addMessage("An error occurred connecting to the backend. Is FastAPI running?", false);
        statusText.textContent = "Error: " + error.message;
        statusText.style.color = 'var(--conf-low-text)';
    } finally {
        sendButton.disabled = false;
        sendButton.style.opacity = '1';
    }
}
