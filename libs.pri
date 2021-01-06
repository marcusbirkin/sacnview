LIBS_PATH = $$system_path($${_PRO_FILE_PWD_}/libs)

# Breakpad
BREAKPAD_PATH = $$system_path($${LIBS_PATH}/breakpad)
DEPOT_TOOLS_PATH = $$system_path($${_PRO_FILE_PWD_}/tools/depot_tools)
INCLUDEPATH += $$system_path($${BREAKPAD_PATH}/src/src)
linux {
    DEFINES += USE_BREAKPAD

    # Fetch breakpad if required
    system([ ! -d $$shell_quote($${BREAKPAD_PATH}) ] && mkdir $$shell_quote($${BREAKPAD_PATH}) && cd $$shell_quote($${BREAKPAD_PATH}) && PATH=$${DEPOT_TOOLS_PATH}:$PATH fetch breakpad)
    # Update Breakpad
    system(cd $$shell_quote($${BREAKPAD_PATH}) && PATH=$${DEPOT_TOOLS_PATH}:$PATH gclient sync)
    # Build
    system(cd $$shell_quote($${BREAKPAD_PATH}/src) && ./configure && make)

    LIBS += -L$${BREAKPAD_PATH}/src/src/client/linux -lbreakpad_client

    HEADERS += src/crash_handler.h \
        src/ui/crash_test.h
    SOURCES += src/crash_handler.cpp \
        src/ui/crash_test.cpp
    FORMS += ui/crash_test.ui
}
win32 {
    DEFINES += USE_BREAKPAD

    # Pull Breakpad dependencies
    system(pushd $$shell_quote($${DEPOT_TOOLS_PATH}) && git reset --hard && git checkout master && git pull && git reset --hard && popd)
    system(cmd /c for %A in ($$shell_quote($${DEPOT_TOOLS_PATH})) do %~sA\update_depot_tools.bat)
    system(cd $${LIBS_PATH} && cmd /c for %A in ($$shell_quote($${DEPOT_TOOLS_PATH})) do %~sA\gclient sync)

    LIBS += -luser32
    INCLUDEPATH  += {BREAKPAD_PATH}/src/
    HEADERS += $${BREAKPAD_PATH}/src/common/windows/string_utils-inl.h \
        $${BREAKPAD_PATH}/src/common/windows/guid_string.h \
        $${BREAKPAD_PATH}/src/client/windows/handler/exception_handler.h \
        $${BREAKPAD_PATH}/src/client/windows/common/ipc_protocol.h \
        $${BREAKPAD_PATH}/src/google_breakpad/common/minidump_format.h \
        $${BREAKPAD_PATH}/src/google_breakpad/common/breakpad_types.h \
        $${BREAKPAD_PATH}/src/client/windows/crash_generation/crash_generation_client.h \
        $${BREAKPAD_PATH}/src/common/scoped_ptr.h \
        src/crash_handler.h \
        src/ui/crash_test.h
    SOURCES += $${BREAKPAD_PATH}/src/client/windows/handler/exception_handler.cc \
        $${BREAKPAD_PATH}/src/common/windows/string_utils.cc \
        $${BREAKPAD_PATH}/src/common/windows/guid_string.cc \
        $${BREAKPAD_PATH}/src/client/windows/crash_generation/crash_generation_client.cc \
        src/crash_handler.cpp \
        src/ui/crash_test.cpp
    FORMS += ui/crash_test.ui
}
macx {
    # Breakpad is disabled for MacOS as it has been superceded by Crashpad
    # see https://groups.google.com/a/chromium.org/forum/#!topic/chromium-dev/6eouc7q2j_g
    DEFINES -= USE_BREAKPAD
}

    # Firewall Checker
win32 {
    LIBS += -lole32 -loleaut32
}

#PCap/WinPcap
win32 {
    # Not for Windows XP Build
    equals(TARGET_WINXP, 0) {
        PCAP_PATH = $${_PRO_FILE_PWD_}/libs/WinPcap-413-173-b4
        contains(QT_ARCH, i386) {
            LIBS += -L$${PCAP_PATH}/lib
        } else {
            LIBS += -L$${PCAP_PATH}/lib/x64
        }
        LIBS += -lwpcap -lPacket
        INCLUDEPATH += $${PCAP_PATH}/Include

        INCLUDEPATH += src/pcap/
        SOURCES += src/pcap/pcapplayback.cpp \
            src/pcap/pcapplaybacksender.cpp
        HEADERS += src/pcap/pcapplayback.h \
            src/pcap/pcapplaybacksender.h \
            src/pcap/ethernetstrut.h
        FORMS += ui/pcapplayback.ui
    }
}
!win32 {
    LIBS += -lpcap

    INCLUDEPATH += src/pcap/
    SOURCES += src/pcap/pcapplayback.cpp \
        src/pcap/pcapplaybacksender.cpp
    HEADERS += src/pcap/pcapplayback.h \
        src/pcap/pcapplaybacksender.h \
        src/pcap/ethernetstrut.h
    FORMS += ui/pcapplayback.ui
}

# OpenSSL
win32 {
    # https://wiki.openssl.org/index.php/Binaries
    equals(QT_MAJOR_VERSION, 5):equals(QT_MINOR_VERSION, 6) { #https://wiki.qt.io/Qt_5.6_Tools_and_Versions
        OPENSSL_VERS = 1.0.2g
    }
    equals(QT_MAJOR_VERSION, 5):equals(QT_MINOR_VERSION, 9) { #https://wiki.qt.io/Qt_5.9_Tools_and_Versions
        OPENSSL_VERS = 1.0.2j
    }
    equals(QT_MAJOR_VERSION, 5):equals(QT_MINOR_VERSION, 10) { #https://wiki.qt.io/Qt_5.10_Tools_and_Versions
        OPENSSL_VERS = 1.0.2j
    }
    equals(QT_MAJOR_VERSION, 5):equals(QT_MINOR_VERSION, 11) { #https://wiki.qt.io/Qt_5.11_Tools_and_Versions
        OPENSSL_VERS = 1.0.2j
    }
    equals(QT_MAJOR_VERSION, 5):equals(QT_MINOR_VERSION, 12) { #https://wiki.qt.io/Qt_5.12_Tools_and_Versions
        OPENSSL_VERS = 1.1.1b
    }
    contains(QT_ARCH, i386) {
        OPENSSL_PATH = $${_PRO_FILE_PWD_}/libs/openssl-$${OPENSSL_VERS}-win32
    } else {
        OPENSSL_PATH = $${_PRO_FILE_PWD_}/libs/openssl-$${OPENSSL_VERS}-win64
    }
}
