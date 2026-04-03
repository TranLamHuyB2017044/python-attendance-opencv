import os
import shutil
import subprocess
import sys
from pathlib import Path
import customtkinter

def build_apps():
    # 1. Setup paths
    project_root = Path(__file__).parent.absolute()
    dist_dir = project_root / "dist_package"
    build_dir = project_root / "build_temp"
    
    # Get customtkinter path for assets (required for themes/icons)
    ctk_path = os.path.dirname(customtkinter.__file__)
    
    # Clean old builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(exist_ok=True)

    print("🚀 Starting Dual-App Build Process (Loader + Source model)...")

    # 2. Build ONE shared runtime (using service_main as base)
    # This creates the '_internal' folder with all libraries (OpenCV, InsightFace, etc.)
    print("📦 Step 1: Building shared runtime and Service Loader...")
    subprocess.run([
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--windowed", # Hidden for service
        "--name", "AttendanceService",
        "--icon", str(project_root / "app_icon.ico") if (project_root / "app_icon.ico").exists() else "NONE",
        "--add-data", f"{ctk_path};customtkinter",
        "--contents-directory", "_internal",
        str(project_root / "src" / "service_main.py")
    ], check=True)

    # Move to our custom dist package
    shutil.move(str(project_root / "dist" / "AttendanceService"), str(dist_dir / "AttendanceSystem"))

    # 3. Build Manager Loader (Sharing the same _internal)
    print("📦 Step 2: Building Manager Loader...")
    subprocess.run([
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "AttendanceManager",
        "--icon", str(project_root / "app_icon.ico") if (project_root / "app_icon.ico").exists() else "NONE",
        "--add-data", f"{ctk_path};customtkinter",
        "--contents-directory", "_internal",
        str(project_root / "src" / "management_app.py")
    ], check=True)

    # Copy ONLY the EXE of Manager to the shared folder
    manager_exe = project_root / "dist" / "AttendanceManager" / "AttendanceManager.exe"
    shutil.copy(str(manager_exe), str(dist_dir / "AttendanceSystem" / "AttendanceManager.exe"))

    # 4. Create 'src' directory in the distribution package
    # This allows updating .py files WITHOUT rebuilding EXEs
    print("📂 Step 3: Preparing Source directory for live updates...")
    target_src = dist_dir / "AttendanceSystem" / "src"
    target_src.mkdir(exist_ok=True)
    
    # Copy all source files
    source_src = project_root / "src"
    for item in source_src.iterdir():
        if item.is_dir():
            if item.name != "__pycache__":
                shutil.copytree(str(item), str(target_src / item.name), dirs_exist_ok=True)
        else:
            shutil.copy2(str(item), str(target_src / item.name))

    # 5. Copy .env.example
    if (project_root / ".env.example").exists():
        shutil.copy2(str(project_root / ".env.example"), str(dist_dir / "AttendanceSystem" / ".env"))

    # 6. Cleanup PyInstaller temp files
    print("🧹 Cleaning up temporary build files...")
    if (project_root / "dist").exists(): shutil.rmtree(project_root / "dist")
    if (project_root / "build").exists(): shutil.rmtree(project_root / "build")
    for spec in project_root.glob("*.spec"): spec.unlink()

    print(f"\n✅ BUILD COMPLETE!")
    print(f"📍 Distribution Folder: {dist_dir / 'AttendanceSystem'}")
    print("\n💡 HOW TO UPDATE FOR CUSTOMERS:")
    print("Just send them the updated files in 'src/' folder.")
    print("They copy and replace them in their installed directory. NO REINSTALL NEEDED.")

if __name__ == "__main__":
    try:
        # Ensure PyInstaller and CustomTkinter are installed
        import PyInstaller
        import customtkinter
        build_apps()
    except ImportError as e:
        print(f"❌ Error: {e.name} not found. Please run 'pip install pyinstaller customtkinter'")
