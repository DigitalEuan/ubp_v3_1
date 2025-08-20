import micropip
import asyncio

async def install_packages():
    # psutil cannot be installed in this environment as it requires a C extension.
    # It has been removed from this list.
    # matplotlib has been added as requested.
    required_packages = ['numpy', 'scipy', 'matplotlib']
    print(f"Attempting to install: {required_packages}...")
    
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            await micropip.install(package)
            print(f"✅ {package} installed successfully.")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
            print("Please ensure your environment allows external package installations.")
            # Continue to try installing other packages even if one fails.

# Run the async installation function
asyncio.run(install_packages())
