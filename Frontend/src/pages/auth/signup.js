import { login, getUser } from '/src/js/auth-guard.js';
import { showToast } from '/src/js/utils.js';

const user = getUser();
if (user) window.location.href = '/src/pages/dashboard/index.html';

const form = document.getElementById('signupForm');
const nameEl = document.getElementById('name');
const emailEl = document.getElementById('email');
const pwdEl = document.getElementById('password');
const confirmEl = document.getElementById('confirmPassword');
const togglePwd = document.getElementById('togglePwd');
const toggleConfirm = document.getElementById('toggleConfirm');
const signupBtn = document.getElementById('signupBtn');
const signupBtnText = document.getElementById('signupBtnText');
const signupSpinner = document.getElementById('signupSpinner');

togglePwd.addEventListener('click', () => {
  pwdEl.type = pwdEl.type === 'password' ? 'text' : 'password';
  togglePwd.textContent = pwdEl.type === 'password' ? '👁️' : '🙈';
});
toggleConfirm.addEventListener('click', () => {
  confirmEl.type = confirmEl.type === 'password' ? 'text' : 'password';
  toggleConfirm.textContent = confirmEl.type === 'password' ? '👁️' : '🙈';
});

// Password strength meter
pwdEl.addEventListener('input', () => {
  const val = pwdEl.value;
  const container = document.getElementById('strengthContainer');
  container.style.display = val ? 'block' : 'none';

  let score = 0;
  if (val.length >= 8) score++;
  if (/[A-Z]/.test(val)) score++;
  if (/[0-9]/.test(val)) score++;
  if (/[^A-Za-z0-9]/.test(val)) score++;

  const colors = ['#ff4757', '#ffb347', '#00c2ff', '#00d26a'];
  const labels = ['Weak', 'Fair', 'Good', 'Strong'];
  document.getElementById('strengthLabel').textContent = `Strength: ${labels[score - 1] || 'Very weak'}`;

  for (let i = 1; i <= 4; i++) {
    const seg = document.getElementById(`seg${i}`);
    seg.style.background = i <= score ? colors[score - 1] : 'var(--bg-600)';
  }
});

function showError(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg; el.style.display = msg ? 'block' : 'none';
}

function validate() {
  let valid = true;
  showError('nameError', ''); showError('emailError', '');
  showError('pwdError', ''); showError('confirmError', '');

  if (nameEl.value.trim().length < 2) { showError('nameError', 'Name must be at least 2 characters'); valid = false; }
  if (!emailEl.value || !/\S+@\S+\.\S+/.test(emailEl.value)) { showError('emailError', 'Valid email required'); valid = false; }
  if (pwdEl.value.length < 8) { showError('pwdError', 'Password must be at least 8 characters'); valid = false; }
  if (pwdEl.value !== confirmEl.value) { showError('confirmError', 'Passwords do not match'); valid = false; }
  if (!document.getElementById('terms').checked) { showToast('Please accept the terms', 'warning'); valid = false; }
  return valid;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!validate()) return;
  signupBtn.disabled = true;
  signupBtnText.textContent = 'Creating account…';
  signupSpinner.style.display = 'inline-flex';
  await new Promise(r => setTimeout(r, 1800));
  login(nameEl.value.trim(), emailEl.value);
  showToast('Account created! Welcome 🎉');
  setTimeout(() => window.location.href = '/src/pages/dashboard/index.html', 700);
});
