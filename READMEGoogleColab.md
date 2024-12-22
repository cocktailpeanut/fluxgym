![Screenshot_2024-12-22_12-42-54](https://github.com/user-attachments/assets/c9629ae7-5e08-408d-b8c2-fc925d2047fa)

currently trying to sort out why a plain testapp file can output the shareable Gradio links.

but adding 'launch(share=True)' doesn´t do anything at all. [edited see my additional comments below]

![Screenshot_2024-12-22_12-44-57](https://github.com/user-attachments/assets/ca55f7d2-4a46-4352-8750-7c870fd734fb)

ah... so the links are still generated. it seems something is causing the links not to appear like what you see above.

unless, you saved your colab into github. so in colab, you would see just the empty space like below screenshot:

![Screenshot_2024-12-22_17-47-38](https://github.com/user-attachments/assets/14aabef1-d364-4c67-80b5-c498080ccaee)

but once you save your copy of the colab notebook into your own Github, the sharelink is actually generated. scroll down to the section below.

![Screenshot_2024-12-22_17-48-44](https://github.com/user-attachments/assets/d68ffe95-5802-4bd1-bbb5-ee7e95726374)

just copy the share link and you're good to go. torch 2.5 seems to work fine so just use that notebook.
