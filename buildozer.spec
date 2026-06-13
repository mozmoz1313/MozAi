[app]
title = MozAi
package.name = mozai
package.domain = org.mozai
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,pth
version = 1.0
requirements = python3,kivy==2.3.0,torch,numpy,yfinance,requests,certifi,urllib3,charset-normalizer,multitasking,peewee
orientation = portrait
fullscreen = 0
android.api = 34
android.minapi = 26
android.ndk = 25b
android.archs = arm64-v8a
android.allow_backup = True
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.gradle_dependencies = 'com.android.support:support-v4:28.0.0'
android.enable_androidx = True
log_level = 2
warn_on_root = 1
[buildozer]
log_level = 2
