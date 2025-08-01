from django import forms

class ScriptInputForm(forms.Form):
    script_text = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'placeholder': 'متن فیلمنامه را وارد کنید'}),
        label="متن فیلمنامه"
    )
