import h2o
import sys
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error")
        sys.exit(2)
    print(sys.argv[1])
    h2o.init(url=f"http://{sys.argv[1]}:54321")
    data = h2o.create_frame(categorical_fraction=0.0,
                              missing_fraction=0.7,
                              rows=6,
                              cols=2,
                              seed=123)
    data.describe()
