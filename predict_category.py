import joblib

def main():
    model = joblib.load("model/product_category_model.pkl")
    print(" Model loaded. Type 'exit' to stop.")

    while True:
        title = input("Enter product title: ").strip()
        if title.lower() == "exit":
            print("Exiting.")
            break

        pred = model.predict([title])[0]
        print(f" Predicted category: {pred}\n" + "-"*40)

if __name__ == "__main__":
    main()
