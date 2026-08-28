# 🥦 VegHealth AI — AI-Based Vegetable Health Monitoring System

A stunning, fully interactive frontend for an AI-powered vegetable health monitoring system. Scan vegetables, check freshness, get nutrition info, plan your diet, and more — all in one app!

---

## 📸 What This App Does

| Feature | Description |
|---|---|
| 🔐 Login / Signup | Create an account or use the Demo login |
| 📷 Camera Scan | Use your webcam to capture vegetables |
| 📁 Gallery Upload | Upload images from your device |
| 📊 Freshness Prediction | AI gauge showing freshness % |
| ⏳ Shelf Life | Days remaining before spoilage |
| 🧪 Nutrition Info | Calories, protein, vitamins per 100g |
| 🛡️ Safety Warning | Safe / Caution / Unsafe banner |
| 💬 AI Chatbot | Ask anything about vegetables |
| 🥗 Diet Planning | 8 health goals × 7-day meal plans |
| 👨‍🍳 Recipes | 9 recipes with search, filter & detail view |
| 📦 Storage Tips | How to store 12 different vegetables |

---

## 🖥️ Requirements

Before starting, make sure your laptop has the following installed:

### 1. Node.js (version 18 or newer)
- Go to: https://nodejs.org
- Download the **LTS** version (recommended)
- Run the installer — click Next → Next → Install
- To **verify installation**, open a terminal and type:
  ```bash
  node --version
  # Should print something like: v20.11.0
  ```

### 2. npm (comes with Node.js automatically)
- Verify it's installed:
  ```bash
  npm --version
  # Should print something like: 10.2.4
  ```

---

## 📂 Getting the Project

### Option A — From a ZIP file (if your friend zipped it)
1. Extract the ZIP file anywhere on your laptop
2. Remember the folder path (e.g., `C:\Projects\VegHealthAI` or `~/Desktop/VegHealthAI`)

### Option B — Copy the folder directly
- Just copy the whole project folder to your laptop

> ⚠️ **Important:** Make sure the folder you received contains a `package.json` file inside it. That's how you know it's the right folder.

---

## 🚀 Running the Project (Step by Step)

### Step 1 — Open a Terminal

**On Windows:**
- Press `Windows + R`, type `cmd`, press Enter
- OR right-click on the Start menu → "Windows Terminal" or "Command Prompt"

**On Mac:**
- Press `Cmd + Space`, type `terminal`, press Enter

---

### Step 2 — Navigate to the Project Folder

Type the following command, replacing the path with wherever you put the project:

```bash
# Windows example:
cd "C:\Projects\AI based vegetable health monitoring sytem"

# Mac/Linux example:
cd ~/Desktop/AI\ based\ vegetable\ health\ monitoring\ sytem
```

> 💡 **Tip:** You can also just drag the project folder into the terminal window — it will auto-fill the path!

---

### Step 3 — Install Dependencies

This downloads all the libraries the project needs. You only need to do this **once**.

```bash
npm install
```

> ⏳ Wait for it to finish — you'll see a progress bar. It may take 1–2 minutes depending on your internet speed.
> When it's done, you'll see something like `added 10 packages` and get back to the prompt.

> ⚠️ If you see a red error like `npm: command not found`, it means Node.js is not installed. Go back to the Requirements section above.

---

### Step 4 — Start the Development Server

```bash
npm run dev
```

You should see output like this:

```
  VITE v8.0.0  ready in 300 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

> ✅ If you see this, the app is running!

---

### Step 5 — Open the App in Your Browser

Open any modern browser (Chrome, Edge, Firefox) and go to:

```
http://localhost:5173
```

> The page will automatically redirect you to the **Login** screen.

---

## 🔑 Logging In

You have two options:

- **Option A — Demo Account:** Click the green **"Try Demo Account"** button. No password needed! Instantly logs you in.
- **Option B — Create Account:** Click "Create one free" → fill in your name, email, and password → you're in!

> 💡 No real server is needed — everything is stored locally in your browser (no data is sent anywhere).

---

## 🛑 Stopping the App

When you're done, go back to the terminal and press:
```
Ctrl + C
```
This stops the development server.

---

## 🔄 Running Again Later

Every time you want to use the app, just:
1. Open terminal
2. `cd` into the project folder
3. Run `npm run dev`
4. Open `http://localhost:5173` in your browser

> You do **not** need to run `npm install` again after the first time.

---

## 📷 Camera Feature Note

The **Camera Scan** feature uses your device's webcam. When you click "Start Camera":
- Your browser will ask for camera permission — click **"Allow"**
- If you're on `localhost`, this works without HTTPS
- If camera doesn't work, use the **Gallery Upload** tab instead and try one of the sample vegetable chips 🍅

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `npm: command not found` | Install Node.js from https://nodejs.org |
| `Port 5173 already in use` | Run `npm run dev -- --port 3000` instead |
| Page shows blank/error | Make sure you typed the URL exactly: `http://localhost:5173` |
| Camera not working | Use "Gallery Upload" tab, or allow camera in browser settings |
| `node_modules not found` | Run `npm install` first |

---

## 📁 Project Structure (for reference)

```
project/
├── package.json          ← Project config (DO NOT delete)
├── vite.config.js        ← Vite build settings
├── index.html            ← Entry point (auto-redirects to login)
├── src/
│   ├── css/
│   │   └── main.css      ← All styles (dark green theme)
│   ├── js/
│   │   ├── auth-guard.js ← Login/session management
│   │   ├── mock-api.js   ← AI simulation + vegetable database
│   │   ├── navbar.js     ← Sidebar navigation component
│   │   └── utils.js      ← Helper functions
│   └── pages/
│       ├── auth/         ← login.html + signup.html
│       ├── dashboard/    ← Home with stats
│       ├── scan/         ← Camera + upload page
│       ├── results/      ← Analysis results
│       ├── chatbot/      ← AI chat assistant
│       ├── diet/         ← Diet planning
│       ├── recipes/      ← Recipe recommendations
│       └── storage/      ← Storage tips
└── node_modules/         ← Auto-installed libraries (DO NOT delete)
```

---

## 🛠️ Tech Stack

- **Vite** — Lightning-fast development server
- **Vanilla JavaScript** — No heavy frameworks
- **Pure CSS** — Dark-mode design system
- **Google Fonts** — Inter + Outfit typography
- **getUserMedia API** — Browser camera access

---

Made with 💚 by VegHealth AI Team
