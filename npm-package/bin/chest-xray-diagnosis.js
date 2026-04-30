#!/usr/bin/env node

/**
 * chest-xray-diagnosis CLI (npx entry point)
 *
 * This thin Node.js wrapper:
 *   1. Checks Python 3.9+ is installed
 *   2. Creates a virtualenv in ~/.chexray-diagnosis/venv/
 *   3. pip installs chest-xray-diagnosis from PyPI
 *   4. Proxies all commands to the Python CLI
 *
 * Usage:
 *   npx chest-xray-diagnosis serve
 *   npx chest-xray-diagnosis analyze path/to/xray.png
 *   npx chest-xray-diagnosis setup
 *   npx chest-xray-diagnosis download-models
 */

const { ensurePythonEnv, runPythonCli } = require('../lib/launcher');

async function main() {
  const args = process.argv.slice(2);

  // --version / --help handled locally for speed
  if (args.includes('--version')) {
    const pkg = require('../package.json');
    console.log(`chest-xray-diagnosis (npm wrapper) v${pkg.version}`);
    // Also try to show Python package version
    try {
      await runPythonCli(['--version']);
    } catch {
      // Python not set up yet — that's fine
    }
    return;
  }

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
╔══════════════════════════════════════════════════════════╗
║        🫁 Chest X-Ray Diagnostic Assistant              ║
║        AI-powered chest X-ray analysis                   ║
╚══════════════════════════════════════════════════════════╝

Commands:
  serve              Start the API server (default port 8000)
  analyze <image>    Analyze a single chest X-ray image
  setup              Configure API keys interactively
  download-models    Pre-download model weights (~230 MB)

Options:
  --help, -h         Show this help message
  --version          Show version

Examples:
  npx chest-xray-diagnosis serve --port 8000
  npx chest-xray-diagnosis analyze chest_xray.png --json
  npx chest-xray-diagnosis setup

Prerequisites:
  • Python 3.9+ (auto-detected)
  • API keys: GEMINI_API_KEY, GEOAPIFY_API_KEY (set via 'setup' command)
`);
    return;
  }

  try {
    console.log('🔍 Checking Python environment...');
    await ensurePythonEnv();
    console.log('✅ Python environment ready\n');

    await runPythonCli(args);
  } catch (err) {
    console.error(`\n❌ ${err.message}`);
    process.exit(1);
  }
}

main();
