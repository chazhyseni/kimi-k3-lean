# Homebrew formula for kimi-k3-lean.
#
# Install via a personal tap:
#   brew tap chazhyseni/kimi-k3-lean https://github.com/chazhyseni/homebrew-kimi-k3-lean
#   brew install kimi-k3-lean
#
# Once installed:
#   k3 --help
#   python3 -m serve /path/to/checkpoint --port 8080 --preset server
#
# This formula is published in the chazhyseni/homebrew-kimi-k3-lean tap.
# The tap repo's only job is to host this file; the binaries it points
# at are the GitHub release artifacts built by .github/workflows/release.yml.

class KimiK3Lean < Formula
  desc "Lean OpenAI-compatible server for Kimi K3 — disk-resident, CPU-only"
  homepage "https://github.com/chazhyseni/kimi-k3-lean"
  url "https://github.com/chazhyseni/kimi-k3-lean/archive/refs/tags/v0.6.8.tar.gz"
  sha256 "PLACEHOLDER_SHA256"  # updated by the release workflow
  license "Apache-2.0"
  head "https://github.com/chazhyseni/kimi-k3-lean.git", branch: "main"

  # Platform support. macOS on Apple Silicon is the primary target; Intel
  # and Linux are supported via the same release artifacts.
  depends_on :macos

  # Build dependencies: gcc/clang + make. Apple Command Line Tools ship both.
  depends_on "make" => :build

  # Runtime: Python 3.11+ for the OpenAI server.
  depends_on "python@3.11"

  def install
    # Build with the Makefile on macOS, passing the LDFLAGS override that
    # conda-free macOS environments don't actually need but the Makefile
    # expects to be passed in.
    system "make", "-j", ENV.make_jobs, "LDFLAGS=-lm -pthread"

    # Install to Homebrew's prefix. The Makefile's install target
    # honours PREFIX; we pass it through.
    system "make", "install", "PREFIX=#{prefix}", "LDFLAGS=-lm -pthread"
  end

  def post_install
    # macOS doesn't have ldconfig; brew handles dylib resolution via
    # the symlinks in $(brew --prefix)/lib, but the kernel-level
    # DYLD_LIBRARY_PATH for one-off launches is occasionally useful.
    # We don't set it permanently — that would interfere with other tools.
    ohai "kimi-k3-lean installed. Run 'k3 --help' to verify."
  end

  test do
    # Smoke-test: --help and --version should both succeed.
    assert_match "Kimi K3 inference engine", shell_output("#{bin}/k3 --version")
    assert_match "usage: k3", shell_output("#{bin}/k3 --help")

    # Verify the shared library exports its public API.
    require "open3"
    out, _err, status = Open3.capture3("nm", "-D", "--defined-only", "#{lib}/libk3.dylib")
    assert status.success?, "nm failed: #{out}"
    %w[k3_open k3_close k3_step k3_generate k3_tokenize k3_detokenize
       k3_save_state k3_load_state k3_get_stats k3_reset_stats
       k3_model_id k3_n_layers k3_vocab_size k3_ctx_size].each do |sym|
      assert_match(/ T #{Regexp.escape(sym)}$/, out, "missing public symbol: #{sym}")
    end
  end
end