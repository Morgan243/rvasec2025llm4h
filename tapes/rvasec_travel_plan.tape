Output ../slides/assets/gifs/agents/rvasec_travel_plan_search_0.gif
Output ../slides/assets/vids/agents/rvasec_travel_plan_search_0.mp4

Set Shell bash
Set Padding 1
Set TypingSpeed 2ms
Set FontSize 30
Set Width 1500
Set Height 800


Hide
Type cd ../src/
Enter
Show

Type 'Q="I need to get from the whole foods in richmond va to the downtown marriot hotel, can you get me some directions by bike?"'
Sleep .5
Enter
Type 'uv --directory . --project ../../EXTERNAL/smolagents/ run smolagent "$Q"  --model-type "LiteLLMModel" --model-id "ollama/qwen2.5-coder:14b-instruct-q4_K_M" --imports "pandas numpy" --tools "web_search" --api-base "http://fractal:11434" --api-key "lolnoproblem"'
Sleep .5
Enter

Enter

#Wait+Screen@60s /Final/
#Wait+Line@15s /\[morgan\@framey/
Wait+Line@120s

# Clear the lines so that the last text is visible when the video is paused
Enter
Enter

