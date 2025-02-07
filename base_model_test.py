from transformers import RobertaTokenizer, T5ForConditionalGeneration


def main():
    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

    # Define Java to Python translation task
    text = """translate Java to Python:
    <JAVA>
    public class Solution {
        public static void main(String[] args) {
            for (int i = 1; i <= 5; i++) {
                System.out.println("Hello World " + i);
            }
        }
    }
    </JAVA>
    """

    # Tokenize input
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    # Generate translation
    generated_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Translated Python Code:")
    print(output)


if __name__ == '__main__':
    main()
