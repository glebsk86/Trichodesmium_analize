import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    from trichodesmium.local_app import main
    raise SystemExit(main())
