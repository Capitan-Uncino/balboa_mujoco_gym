{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.mujoco
      python-pkgs.numpy 
      python-pkgs.scipy
    ]))
    pkgs.libdecor      
  ];

  # FIX 2: Explicitly point to the libraries MuJoCo needs at runtime
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    libGL
    glfw
    wayland
    libxkbcommon
    libdecor           
    libX11
    libXrandr
    libXi
    libXcursor
    libXext
    libXinerama
  ]);
  
  # FIX 3: Force MuJoCo to use the GLFW backend explicitly to avoid EGL ambiguity
  MUJOCO_GL = "glfw"; 

  shellHook = ''
    echo "========================================="
    echo "🤖 MuJoCo Balboa Environment (Patched)"
    echo "========================================="
    # Should fix the 'Invalid value (0) for DRI_PRIME' on some hybrid GPU laptops
    export DRI_PRIME=0 
  '';
}
