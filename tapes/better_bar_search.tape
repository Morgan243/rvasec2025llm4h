Output ../slides/assets/tape_outputs/better_bar_search_0.gif
Output ../slides/assets/tape_outputs/better_bar_search_0.mp4

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

Type 'M="gemma3:12b"'
Enter

Type 'Q="Find some good bars for the after after party at rvasec 14 this year. Include a short description of each."'
Sleep 1.5
Enter
Type 'uv --directory . --project ../../EXTERNAL/smolagents/ run smolagents_better_search.py --query "$Q" --api_base "http://fractal:11434"  --model_id="ollama/$M"'
Enter

Wait+Line@240s
#Sleep 60

# Clear the lines so that the last text is visible when the video is paused
Enter
Enter
