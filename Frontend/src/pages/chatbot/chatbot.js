import { requireAuth } from '/src/js/auth-guard.js';
import { injectSidebar } from '/src/js/navbar.js';
import { getChatResponse } from '/src/js/mock-api.js';

requireAuth();
injectSidebar('chatbot');

const messagesEl = document.getElementById('chatMessages');
const inputEl = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

function appendMessage(role, text) {
  const isUser = role === 'user';
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="message-avatar">${isUser ? '🧑' : '🤖'}</div>
    <div class="message-bubble">${text}</div>
  `;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'message bot';
  div.id = 'typingIndicator';
  div.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-bubble">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTyping() {
  document.getElementById('typingIndicator')?.remove();
}

async function sendMessage(text) {
  if (!text.trim()) return;
  appendMessage('user', text);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  showTyping();
  const response = await getChatResponse(text);
  removeTyping();
  appendMessage('bot', response);
  sendBtn.disabled = false;
}

// Initial welcome message
appendMessage('bot', '👋 Hello! I\'m <strong>VegBot</strong>, your AI vegetable health assistant.<br><br>I can help you with:<br>• 🌡️ Freshness detection tips<br>• 🧪 Nutrition information<br>• 📦 Storage advice<br>• 👨‍🍳 Recipe suggestions<br><br>What would you like to know?');

// Quick chips
document.getElementById('quickChips').querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => sendMessage(chip.dataset.msg));
});

// Send button
sendBtn.addEventListener('click', () => sendMessage(inputEl.value));
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(inputEl.value);
  }
});

// Auto-resize textarea
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});
