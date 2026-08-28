// Auth guard - redirects unauthenticated users to login
export function requireAuth() {
  const user = getUser();
  if (!user) {
    window.location.href = '/src/pages/auth/login.html';
    return null;
  }
  return user;
}

export function getUser() {
  const data = localStorage.getItem('veghealth_user');
  return data ? JSON.parse(data) : null;
}

export function logout() {
  localStorage.removeItem('veghealth_user');
  window.location.href = '/src/pages/auth/login.html';
}

export function login(name, email) {
  const user = { name, email, id: Date.now() };
  localStorage.setItem('veghealth_user', JSON.stringify(user));
  return user;
}
