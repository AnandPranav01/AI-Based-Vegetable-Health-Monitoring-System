import { login, getUser } from '/src/js/auth-guard.js';
import { showToast } from '/src/js/utils.js';

const user = getUser();
if (user) window.location.href = '/src/pages/dashboard/index.html';

const form = document.getElementById('loginForm');
const emailEl = document.getElementById('email');
const pwdEl = document.getElementById('password');
const togglePwd = document.getElementById('togglePwd');
const loginBtn = document.getElementById('loginBtn');
const loginBtnText = document.getElementById('loginBtnText');
const loginSpinner = document.getElementById('loginSpinner');
const demoBtn = document.getElementById('demoBtn');

togglePwd.addEventListener('click', () => {
  pwdEl.type = pwdEl.type === 'password' ? 'text' : 'password';
  togglePwd.textContent = pwdEl.type === 'password' ? '👁️' : '🙈';
});

function setLoading(loading) {
  loginBtn.disabled = loading;
  loginBtnText.textContent = loading ? 'Signing in...' : 'Sign In';
  loginSpinner.style.display = loading ? 'inline-flex' : 'none';
}

function validate() {
  let valid = true;
  const emailError = document.getElementById('emailError');
  const pwdError = document.getElementById('pwdError');

  emailError.style.display = 'none';
  pwdError.style.display = 'none';

  if (!emailEl.value || !/\S+@\S+\.\S+/.test(emailEl.value)) {
    emailError.textContent = 'Please enter a valid email address';
    emailError.style.display = 'block';
    valid = false;
  }
  if (pwdEl.value.length < 6) {
    pwdError.textContent = 'Password must be at least 6 characters';
    pwdError.style.display = 'block';
    valid = false;
  }
  return valid;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!validate()) return;
  setLoading(true);
  await new Promise(r => setTimeout(r, 1500));
  const name = emailEl.value.split('@')[0].replace(/[._]/g, ' ');
  login(name.charAt(0).toUpperCase() + name.slice(1), emailEl.value);
  showToast('Signed in successfully! 🎉');
  setTimeout(() => window.location.href = '/src/pages/dashboard/index.html', 800);
});

demoBtn.addEventListener('click', async () => {
  demoBtn.disabled = true;
  demoBtn.textContent = '🚀 Loading demo…';
  await new Promise(r => setTimeout(r, 1000));
  login('Demo User', 'demo@veghealth.ai');
  showToast('Welcome to the demo! 🌿');
  setTimeout(() => window.location.href = '/src/pages/dashboard/index.html', 600);
});
