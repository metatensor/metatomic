#!/usr/bin/env nu

def main [] {
    let root = ($env.PWD)
    let deps_dir = ($root | path join "docs/subprojects")
    let doxyrest_src = ($deps_dir | path join "doxyrest")
    let install_dir = ($deps_dir | path join "install")
    let has_doxyrest = (which doxyrest | is-not-empty)

    # Setup Directory Structure
    if not ($deps_dir | path exists) {
        mkdir $deps_dir
    }

    # Grab Doxyrest if missing
    if not $has_doxyrest and not ($install_dir | path exists) {
        print "Downloading Doxyrest binary release..."
        let version = "2.1.3"
        let target = "linux-amd64"
        let archive = $"doxyrest-($version)-($target).tar.xz"
        let url = $"https://github.com/vovkos/doxyrest/releases/download/doxyrest-($version)/($archive)"

        cd $deps_dir
        http get $url | save $archive
        tar -xJf $archive
        
       if ($install_dir | path exists) { rm -rf $install_dir }
        mv $"doxyrest-($version)-($target)" $install_dir
        
        rm $archive
        cd $root
    }

    # Still needed for the frames..
    if not ($doxyrest_src | path exists) {
        print $"Cloning doxyrest into ($doxyrest_src)..."
        git clone --depth 1 --single-branch https://github.com/vovkos/doxyrest $doxyrest_src
    }

    # Run the Pipeline
    print "Generating XML with Doxygen..."
    cd docs
    mkdir build/doxygen/xml
    doxygen Doxyfile.metatomic

    print "Generating RST with local Doxyrest..."
    # Add local bin to path for this execution
    with-env { PATH: ($env.PATH | prepend ($install_dir | path join "bin")) } {
        doxyrest -c doxyrest-config.lua
    }

    print "Building HTML with Sphinx (tox)..."
    tox -e docs
}
