#!/bin/bash
set -e

echo "=================================================="
echo "    REBUILDING PYTHON 3.10 WITH ALL MODULES"
echo "=================================================="
echo ""

# Install all required development libraries
echo "📦 Installing required development libraries..."
apt-get update -qq
apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    liblzma-dev \
    uuid-dev \
    tk-dev \
    zlib1g-dev \
    libdb-dev \
    wget

# Set Python version and paths
PYTHON_VERSION="3.10.13"
PYTHON_INSTALL_PREFIX="/storage/python_versions/python3.10"
PYTHON_SOURCE_DIR="/tmp/python_build"

# Clean up any previous build
rm -rf "$PYTHON_SOURCE_DIR"
mkdir -p "$PYTHON_SOURCE_DIR"
cd "$PYTHON_SOURCE_DIR"

# Download Python source
echo "📥 Downloading Python ${PYTHON_VERSION} source..."
wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
tar -xzf "Python-${PYTHON_VERSION}.tgz"
cd "Python-${PYTHON_VERSION}"

# Configure with all modules enabled
echo "⚙️  Configuring Python build..."
./configure \
    --prefix="$PYTHON_INSTALL_PREFIX" \
    --enable-optimizations \
    --enable-shared \
    --with-lto \
    --with-system-ffi \
    --with-computed-gotos \
    --enable-loadable-sqlite-extensions \
    LDFLAGS="-Wl,-rpath,$PYTHON_INSTALL_PREFIX/lib"

# Build Python
echo "🔨 Building Python (this will take a while)..."
make -j$(nproc)

# Backup old installation if it exists
if [ -d "$PYTHON_INSTALL_PREFIX" ]; then
    echo "📦 Backing up old Python installation..."
    mv "$PYTHON_INSTALL_PREFIX" "${PYTHON_INSTALL_PREFIX}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Install Python
echo "📥 Installing Python to $PYTHON_INSTALL_PREFIX..."
make install

# Set up library path
export LD_LIBRARY_PATH="$PYTHON_INSTALL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "$PYTHON_INSTALL_PREFIX/lib" | tee /etc/ld.so.conf.d/python3.10.conf > /dev/null
ldconfig

# Create symlinks
echo "🔗 Creating symlinks..."
ln -sf "$PYTHON_INSTALL_PREFIX/bin/python3.10" /usr/local/bin/python3.10
ln -sf "$PYTHON_INSTALL_PREFIX/bin/python3.10" /usr/local/bin/python3
ln -sf "$PYTHON_INSTALL_PREFIX/bin/pip3.10" /usr/local/bin/pip3.10
ln -sf "$PYTHON_INSTALL_PREFIX/bin/pip3.10" /usr/local/bin/pip3

# Update PATH
export PATH="$PYTHON_INSTALL_PREFIX/bin:$PATH"

# Verify installation
echo ""
echo "=================================================="
echo "           VERIFYING PYTHON INSTALLATION"
echo "=================================================="
echo ""

echo "Python version:"
"$PYTHON_INSTALL_PREFIX/bin/python3.10" --version

echo ""
echo "Testing critical modules..."
"$PYTHON_INSTALL_PREFIX/bin/python3.10" << 'PYTHON_VERIFY'
import sys
modules_to_test = [
    '_bz2', '_lzma', '_ssl', 'ssl', '_sqlite3', 'sqlite3',
    '_ctypes', '_hashlib', '_json', '_multiprocessing', '_pickle', 
    '_socket', '_struct', '_uuid', 'zlib'
]

missing = []
for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"✅ {mod}")
    except ImportError as e:
        print(f"❌ {mod} - MISSING!")
        missing.append(mod)

if missing:
    print(f"\n⚠️  Missing modules: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\n✅ ALL CRITICAL MODULES AVAILABLE!")
PYTHON_VERIFY

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "    ✅ PYTHON 3.10 REBUILT SUCCESSFULLY!"
    echo "=================================================="
    echo ""
    echo "Python is now installed at: $PYTHON_INSTALL_PREFIX"
    echo "You can now run the main.sh script again."
    
    # Clean up build directory
    echo "🧹 Cleaning up build directory..."
    rm -rf "$PYTHON_SOURCE_DIR"
else
    echo ""
    echo "=================================================="
    echo "    ❌ PYTHON VERIFICATION FAILED"
    echo "=================================================="
    echo ""
    echo "Some modules are still missing. Please check the build logs."
    exit 1
fi

