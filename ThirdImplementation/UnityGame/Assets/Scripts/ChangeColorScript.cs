using UnityEngine;

public class ChangeColorScript : MonoBehaviour
{
    SpriteRenderer renderer;
    public Color newColor;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        GameManagerScript.alarmInitiated += ChangeColor;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void ChangeColor()
    {
        renderer.color = newColor;
    }
}
