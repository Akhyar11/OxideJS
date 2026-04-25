# Installation and Setup Guide

Follow the steps below to set up the ML-V1 development environment on your local machine.

## Main Prerequisites

Before starting, ensure you have the following software installed:
1. **Node.js**: LTS version (v20.x or newer recommended).
2. **Rust Toolchain**: Required to build the native backend (`cargo`, `rustc`). Install via [rustup.rs](https://rustup.rs/).
3. **TypeScript**: This framework uses TypeScript for core development.

---

## Installation Steps

### 1. Clone the Repository
If you haven't already, clone this repository to your local machine:
```bash
git clone <repository-url>
cd ML-V1
```

### 2. Install Node.js Dependencies
Use npm to install all required packages:
```bash
npm install
```

### 3. Build Native Backend (Optional but Recommended)
To achieve maximum performance, you need to build the Rust module using `napi-rs`.

**For Production Build (Release):**
```bash
npm run build:rust
```

**For Debug Build (Faster, Lower Performance):**
```bash
npm run build:rust:debug
```

The build process will produce a `.node` binary file in the project root (e.g., `ml-native.linux-x64-gnu.node`).

---

## Installation Verification

After installation is complete, you can verify if the native backend is active by running a simple test:

```ts
import { isNativeAvailable } from "./src/math/rust_backend";

console.log("Native Backend Status:", isNativeAvailable());
```

Alternatively, run the full test suite:
```bash
npm test
```

Notes:
- `npm test` runs a single entry point `test/index.ts`.
- That entry point calls two suites: `test/correctness` for contract/behavior tests and `test/benchmark` for synthetic benchmarks.

---

## Environment Configuration

### Forcibly Disabling the Native Backend
If you encounter issues with the native binary or want to perform debugging on the pure JavaScript implementation, you can use the `ML_DISABLE_NATIVE` environment variable:

```bash
ML_DISABLE_NATIVE=1 node your-script.js
```

### TypeScript Usage
Since this project uses TypeScript extensively, you may need to compile before running scripts with pure `node`:
```bash
npm run build # Runs tsc
```
Alternatively, use `ts-node` for direct execution (already included in `devDependencies`).

---

> [!WARNING]
> If you change your operating system or CPU architecture (e.g., from Linux to macOS), you **must** rerun `npm run build:rust` so that the binary matches the new platform.

**Next Steps:**
Learn how to use this library in the [Quick Start Tutorial](03-tutorial.md).
