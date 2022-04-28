# Based on https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
import tweepy
import torch
import requests
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

loc = "nlpconnect/vit-gpt2-image-captioning"  # vision encoder-decoder model
feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_image(image_url):
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    return pixel_values


def generate_step(pixel_values):
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == "__main__":
    # authentication tokens must be known beforehand - change here
    auth = tweepy.OAuth1UserHandler("123456789", "abcdefg", "123456789", "abcdefg")
        
    print("Authenticate")
    api = tweepy.API(auth)

    recent_tweets = api.user_timeline(screen_name='BoschBot', exclude_replies=True, include_rts=False, count=25)  # exclude replies and retweets
    for t in recent_tweets:
        if 'media' in t.entities:  # only post containing images
            for image in t.entities['media']:
                print(image['media_url'])
                
                img = load_image(image['media_url'])
                pred_caption = generate_step(img)
                caption = f'Caption: {pred_caption[0]}'
                print(caption)
     
                # Get the tweet you want to quote
                tweet_to_quote_url = f"https://twitter.com/BoschBot/status/{t.id}"
    
                # Quote it in a new status
                api.update_status(caption, attachment_url=tweet_to_quote_url)   
