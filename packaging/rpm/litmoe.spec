# Spec file for litMoE RPM (Fedora / RHEL / openSUSE).
#
# Build with:
#   rpmbuild -ba litMoE.spec
#
# Produces:
#   ~/rpmbuild/RPMS/x86_64/litMoE-0.6.8-1.fc39.x86_64.rpm
#
# For .deb on Debian/Ubuntu, use `alien --to-deb` on the resulting RPM,
# or build with checkinstall as documented in packaging/README.md.

Name:           litMoE
Version:        0.6.8
Release:        1%{?dist}
Summary:        Lean OpenAI-compatible server for Kimi K3 — disk-resident, CPU-only
License:        Apache-2.0
URL:            https://github.com/chazhyseni/litMoE
Source0:        https://github.com/chazhyseni/litMoE/archive/refs/tags/v%{version}.tar.gz

BuildArch:      x86_64
BuildRequires:  gcc
BuildRequires:  make
BuildRequires:  python3-devel
BuildRequires:  python3-pip

Requires:       python3 >= 3.11
Requires:       libgomp

%global debug_package %{nil}

%description
litMoE is a lean OpenAI-compatible HTTP server for the Kimi K3
inference engine. The engine runs the full 2.78-trillion-parameter Kimi
K3 on CPU hardware with an 8 GB RAM floor and disk-resident experts.
The server is a thin Python layer over a C library (liblitmoe.so) and
exposes the standard OpenAI Chat Completions API at /v1/chat/completions.

This package contains:
  - /usr/bin/litmoe                 — CLI binary
  - /usr/lib64/liblitmoe.so         — shared library
  - /usr/include/liblitmoe/liblitmoe.h  — public C API
  - /usr/share/doc/litMoE/ — README, LICENSE, INSTALL.md

%prep
%autosetup -n litMoE-%{version}

%build
# The article's Makefile uses LDFLAGS ?= which doesn't append on
# systems where conda or another env sets LDFLAGS. Always pass it.
export LDFLAGS="-lm -pthread"
make -j%{?_smp_build_ncpus} %{?_smp_mflags}

%install
export LDFLAGS="-lm -pthread"
make install DESTDIR=%{buildroot} PREFIX=%{_prefix}

# Move docs into the rpm-doc convention.
install -d %{buildroot}%{_docdir}/litMoE
cp README.md docs/INSTALL.md \
   %{buildroot}%{_docdir}/litMoE/ 2>/dev/null || true
cp LICENSE %{buildroot}%{_docdir}/litMoE/

%files
%license LICENSE
%doc %{_docdir}/litMoE/README.md
%doc %{_docdir}/litMoE/INSTALL.md
%{_bindir}/k3
%{_libdir}/liblitmoe.so
%{_libdir}/liblitmoe_static.a
%{_includedir}/liblitmoe/liblitmoe.h
%{_docdir}/litMoE/

%post
# ldconfig after install.
ldconfig

%postun
ldconfig

%changelog
* Fri Aug 14 2026 chazhyseni <dev@litMoE.local> - 0.6.8-1
- Initial RPM release