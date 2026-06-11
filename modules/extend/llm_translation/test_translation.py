from llm_translation import translate_with_llm

def run_test():
    # 10 clinical English medical terms to test
    test_samples = [
        "Myocardial infarction",
        "Chronic obstructive pulmonary disease",
        "Major depressive disorder",
        "Rheumatoid arthritis",
        "Acute kidney injury",
        "Tuberculosis",
        "Asthma",
        "Atrial fibrillation",
        "Sepsis",
        "Peptic ulcer disease"
    ]

    output_file = "test_results.txt"

    print("=== Starting vLLM Translation Test ===")
    print(f"Testing 10 clinical samples. Results will be saved to {output_file}...\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== vLLM Translation Test Results ===\n\n")
        for i, term in enumerate(test_samples, 1):
            print(f"Sample {i}/10: '{term}'")
            
            # Call the translation function
            translation = translate_with_llm(term)
            
            if translation:
                # Note: debug statements inside llm_translation.py will print to terminal too
                print(f"Final Result: '{translation}'\n")
                f.write(f"Sample {i}: '{term}' -> '{translation}'\n")
            else:
                print("Result: [FAILED or EMPTY RESPONSE]\n")
                f.write(f"Sample {i}: '{term}' -> [FAILED or EMPTY RESPONSE]\n")

    print(f"=== Test Complete. Results saved to {output_file} ===")

if __name__ == "__main__":
    run_test()
