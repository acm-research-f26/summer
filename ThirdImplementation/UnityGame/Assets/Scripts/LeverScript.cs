using System;
using UnityEngine;

public class LeverScript : MonoBehaviour
{
    public Sprite onSprite;
    SpriteRenderer renderer;
    public static Action leverPressed;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "Player")
        {
            renderer.sprite = onSprite;
            GameManagerScript.instance.leverPressed = true;
            leverPressed.Invoke();
        }
    }
}
