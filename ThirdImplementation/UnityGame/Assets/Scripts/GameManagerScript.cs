using UnityEngine;

public class GameManagerScript : MonoBehaviour
{
    public static GameManagerScript instance;
    public bool hasHammer;
    public bool leverPressed;
    public bool diamondStolen;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        hasHammer = false;
        instance = this;
        diamondStolen = false;
        leverPressed = false;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
