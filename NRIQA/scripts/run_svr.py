from nriqa.config import FEATURE_DIR, SVR_CFG
from nriqa.quality.svr_fusion import run_multilayer_svr

def main():
    result = run_multilayer_svr(FEATURE_DIR, SVR_CFG)
    print(f"RMSE = {result['final_rmse']:.4f}")
    print(f"PLCC = {result['final_plcc']:.4f}")
    print(f"SRCC = {result['final_srcc']:.4f}")

    print("\nFirst 10 predictions:")
    for i in range(min(10, len(result["y_test"]))):
        print(f"{i:02d} | true={result['y_test'][i]:.4f} | pred={result['final_pred'][i]:.4f}")


if __name__ == "__main__":
    main()
