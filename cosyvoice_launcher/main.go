package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type Config struct {
	Port     int
	ModelDir string
	LogFile  string
	Timeout  time.Duration
}

func main() {
	if runtime.GOOS != "windows" {
		fmt.Println("[ERROR] This launcher is intended for Windows.")
	}

	cfg := Config{}
	flag.IntVar(&cfg.Port, "port", 39841, "Port to run the CosyVoice Web UI on")
	flag.StringVar(&cfg.ModelDir, "model-dir", filepath.FromSlash("pretrained_models/CosyVoice2-0.5B"), "Model directory (local path or ModelScope id)")
	flag.StringVar(&cfg.LogFile, "log-file", filepath.FromSlash("cosyvoice_webui.log"), "File to write server logs")
	flag.DurationVar(&cfg.Timeout, "wait", 90*time.Second, "How long to wait for the Web UI to become reachable")
	flag.Parse()

	wd, err := os.Getwd()
	fatalIf(err)
	// If launched from the module directory (cosyvoice_launcher), use its parent as repo root.
	repoRoot := wd
	if strings.EqualFold(filepath.Base(wd), "cosyvoice_launcher") {
		repoRoot = filepath.Dir(wd)
	}

	url := fmt.Sprintf("http://localhost:%d/", cfg.Port)
	waitURL := fmt.Sprintf("http://127.0.0.1:%d/", cfg.Port)

	ctx := context.Background()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)

	fmt.Println("[1/4] Preparing Python environment (.venv + deps)...")
	pythonExe, err := ensureVenvAndDeps(ctx, repoRoot)
	fatalIf(err)

	// Start webui.py; keep everything in ONE console by redirecting server logs to a file.
	logPath := cfg.LogFile
	if !filepath.IsAbs(logPath) {
		logPath = filepath.Join(repoRoot, logPath)
	}
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	fatalIf(err)
	defer logFile.Close()

	fmt.Println("[2/4] Starting CosyVoice Web UI...")
	webuiCmd, err := startWebUI(repoRoot, pythonExe, cfg, logFile)
	fatalIf(err)

	fmt.Println("[3/4] Waiting for Web UI to be reachable...")
	err = waitForURL(waitURL, cfg.Timeout)
	if err != nil {
		fmt.Printf("[ERROR] Web UI did not become reachable: %v\n", err)
		fmt.Printf("Check logs: %s\n", logPath)
		fmt.Println("The server process may still be starting; not killing it.")
		nonClosingExit(ctx, 1)
		return
	}

	fmt.Println()
	fmt.Println("============================================================")
	fmt.Println("CosyVoice Web UI")
	fmt.Println("Open in your browser:")
	fmt.Println("  " + url)
	fmt.Println()
	fmt.Println("To stop: press Ctrl+C")
	fmt.Println("============================================================")

	// Wait for Ctrl+C, then stop the webui process and exit.
	<-sigCh
	fmt.Println("Stopping...")
	if webuiCmd != nil && webuiCmd.Process != nil {
		_ = webuiCmd.Process.Kill()
	}
	os.Exit(0)
}

func fatalIf(err error) {
	if err != nil {
		fmt.Printf("[FATAL] %v\n", err)
		time.Sleep(100 * time.Millisecond)
		os.Exit(1)
	}
}

func ensureVenvAndDeps(ctx context.Context, repoRoot string) (string, error) {
	venvPython := filepath.Join(repoRoot, ".venv", "Scripts", "python.exe")
	if _, err := os.Stat(venvPython); errors.Is(err, os.ErrNotExist) {
		// Create venv using whatever python is on PATH.
		if err := runConsole(ctx, repoRoot, "python", []string{"-m", "venv", ".venv"}); err != nil {
			return "", fmt.Errorf("create venv failed: %w", err)
		}
	}

	// Ensure pip exists inside venv.
	if err := runConsole(ctx, repoRoot, venvPython, []string{"-m", "pip", "--version"}); err != nil {
		if err := runConsole(ctx, repoRoot, venvPython, []string{"-m", "ensurepip", "--upgrade"}); err != nil {
			return "", fmt.Errorf("ensurepip failed: %w", err)
		}
	}

	// Upgrade pip.
	_ = runConsole(ctx, repoRoot, venvPython, []string{"-m", "pip", "install", "--upgrade", "pip"})

	// Install runtime deps (includes gradio, HyperPyYAML, and a ruamel.yaml pin).
	req := filepath.Join(repoRoot, "requirements.windows.runtime.txt")
	if err := runConsole(ctx, repoRoot, venvPython, []string{"-m", "pip", "install", "-r", req}); err != nil {
		return "", fmt.Errorf("pip install requirements failed: %w", err)
	}

	// Extra safety: enforce ruamel.yaml pin (fixes HyperPyYAML max_depth crash).
	_ = runConsole(ctx, repoRoot, venvPython, []string{"-m", "pip", "install", "ruamel.yaml==0.18.16"})

	return venvPython, nil
}

func startWebUI(repoRoot, pythonExe string, cfg Config, logFile *os.File) (*exec.Cmd, error) {
	args := []string{"webui.py", "--port", fmt.Sprintf("%d", cfg.Port), "--model_dir", cfg.ModelDir, "--simple"}
	cmd := exec.Command(pythonExe, args...)
	cmd.Dir = repoRoot

	// Ensure Matcha-TTS submodule is importable.
	env := os.Environ()
	matcha := filepath.Join(repoRoot, "third_party", "Matcha-TTS")
	env = prependEnvPath(env, "PYTHONPATH", matcha)
	cmd.Env = env

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func waitForURL(url string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	transport := &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout:   250 * time.Millisecond,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}
	client := &http.Client{Timeout: 800 * time.Millisecond, Transport: transport}
	for {
		if time.Now().After(deadline) {
			return fmt.Errorf("timeout after %s", timeout)
		}
		req, _ := http.NewRequest(http.MethodHead, url, nil)
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 500 {
				return nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func run(ctx context.Context, dir, exe string, args []string, stdout, stderr *os.File) error {
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Dir = dir
	if stdout != nil {
		cmd.Stdout = stdout
	}
	if stderr != nil {
		cmd.Stderr = stderr
	}
	return cmd.Run()
}

func runConsole(ctx context.Context, dir, exe string, args []string) error {
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Dir = dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func prependEnvPath(env []string, key, value string) []string {
	keyEq := key + "="
	for i, kv := range env {
		if strings.HasPrefix(strings.ToUpper(kv), strings.ToUpper(keyEq)) {
			cur := strings.TrimPrefix(kv, keyEq)
			if cur == "" {
				env[i] = keyEq + value
			} else {
				env[i] = keyEq + value + string(os.PathListSeparator) + cur
			}
			return env
		}
	}
	return append(env, keyEq+value)
}

func nonClosingExit(ctx context.Context, code int) {
	_ = ctx
	_ = code
	for {
		time.Sleep(24 * time.Hour)
	}
}
