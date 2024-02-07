#include <iostream>
#include <string>

class String :
    public std::string
{};

void determineSiriusExecutable(String executable)
{ 
    // if executable was not provided
    if (executable.empty())
    {
        std::cout << "Checkpoint #1\n" << std::endl;
    }
    std::cout << "Here" << std::endl;
    std::cout << executable << std::endl;
}

int main()
{
    determineSiriusExecutable("C:\\Program Files\\OpenMS-3.1.0\\share\\OpenMS\\THIRDPARTY\\Sirius\\sirius.bat");
}