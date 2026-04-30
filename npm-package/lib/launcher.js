/**
 * Python environment launcher for chest-xray-diagnosis.
 *
 * Handles:
 *   - Detecting Python 3.9+
 *   - Creating/reusing a virtualenv at ~/.chexray-diagnosis/venv/
 *   - Installing the chest-xray-diagnosis pip package
 *   - Running CLI commands inside that venv
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const CACHE_DIR = path.join(os.homedir(), '.chexray-diagnosis');
const VENV_DIR = path.join(CACHE_DIR, 'venv');
const PIP_PACKAGE = 'chest-xray-diagnosis';

// ---------- helpers ----------

function which(cmd) {
  try {
    const result = execSync(
      process.platform === 'win32' ? `where ${cmd}` : `which ${cmd}`,
      { stdio: 'pipe', encoding: 'utf-8' }
    );
    return result.trim().split(/\r?\n/)[0];
  } catch {
    return null;
  }
}

function getPythonCmd() {
  // Try python3 first, then python
  for (const cmd of ['python3', 'python']) {
    const p = which(cmd);
    if (!p) continue;
    try {
      const ver = execSync(`${cmd} --version`, { stdio: 'pipe', encoding: 'utf-8' }).trim();
      const match = ver.match(/Python (\d+)\.(\d+)/);
      if (match) {
        const major = parseInt(match[1], 10);
        const minor = parseInt(match[2], 10);
        if (major >= 3 && minor >= 9) {
          return cmd;
        }
      }
    } catch { /* continue */ }
  }
  return null;
}

function getVenvPython() {
  return process.platform === 'win32'
    ? path.join(VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VENV_DIR, 'bin', 'python');
}

function getVenvPip() {
  return process.platform === 'win32'
    ? path.join(VENV_DIR, 'Scripts', 'pip.exe')
    : path.join(VENV_DIR, 'bin', 'pip');
}

function venvExists() {
  return fs.existsSync(getVenvPython());
}

function packageInstalled() {
  try {
    execSync(`"${getVenvPip()}" show ${PIP_PACKAGE}`, { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

// ---------- public API ----------

async function ensurePythonEnv() {
  // 1. Check Python
  const pythonCmd = getPythonCmd();
  if (!pythonCmd) {
    throw new Error(
      'Python 3.9+ is required but not found.\n' +
      'Install Python from https://www.python.org/downloads/'
    );
  }
  console.log(`  Python: ${execSync(`${pythonCmd} --version`, { encoding: 'utf-8' }).trim()}`);

  // 2. Create venv if needed
  if (!venvExists()) {
    console.log(`  Creating virtualenv at ${VENV_DIR}...`);
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    execSync(`${pythonCmd} -m venv "${VENV_DIR}"`, { stdio: 'inherit' });
  } else {
    console.log(`  Virtualenv: ${VENV_DIR} (cached)`);
  }

  // 3. Install/upgrade pip package
  if (!packageInstalled()) {
    console.log(`  Installing ${PIP_PACKAGE} from PyPI...`);
    execSync(`"${getVenvPip()}" install --upgrade ${PIP_PACKAGE}`, { stdio: 'inherit' });
  } else {
    console.log(`  Package: ${PIP_PACKAGE} (already installed)`);
  }
}

function runPythonCli(args) {
  return new Promise((resolve, reject) => {
    const venvPython = getVenvPython();
    const child = spawn(venvPython, ['-m', 'chexray_diagnosis.cli', ...args], {
      stdio: 'inherit',
      env: { ...process.env },
    });

    child.on('error', (err) => reject(new Error(`Failed to start Python: ${err.message}`)));
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Python process exited with code ${code}`));
    });
  });
}

module.exports = { ensurePythonEnv, runPythonCli };
